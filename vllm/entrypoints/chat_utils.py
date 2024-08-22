import codecs
from dataclasses import dataclass
from functools import lru_cache
from typing import (Awaitable, Iterable, List, Literal, Optional, Tuple, Union,
                    cast, final)

import pandas as pd
# yapf conflicts with isort for this block
# yapf: disable
from openai.types.chat import ChatCompletionContentPartImageParam
from openai.types.chat import (
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam)
from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat import (
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam)
# yapf: enable
# pydantic needs the TypedDict from typing_extensions
from pydantic import ConfigDict
from transformers import PreTrainedTokenizer
from typing_extensions import Required, TypedDict

from vllm.config import ModelConfig
from vllm.envs import VLLM_TABLE_INSERT_EMBS_TOKEN, VLLM_TABLE_INSERT_SEP_TOKEN
from vllm.logger import init_logger
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.utils import async_get_and_parse_image

logger = init_logger(__name__)


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


class TableCols(TypedDict, total=False):
    name: Required[str]
    dtype: Required[str]
    unique: Required[bool]
    contains_nan: Required[bool]
    
    comment: Optional[str] = None

class Table(TypedDict, total=False):
    varname: Required[str]
    
    cols: Required[List[TableCols]]

    # TODO: validate the length, can check the max length about the input csv value
    values: Required[List[List]]


class ChatCompletionContentPartTableParam(TypedDict, total=False):
    table: Required[List[Table]]

    type: Required[Literal["table"]]
    """The type of the content part."""


ChatCompletionContentPartParam = Union[OpenAIChatCompletionContentPartParam,
                                       ChatCompletionContentPartTableParam,
                                       CustomChatCompletionContentPartParam]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""
    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam]]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """


ChatCompletionMessageParam = Union[OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam]


@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]
    mm_futures: List[Awaitable[MultiModalDataDict]]


def load_chat_template(chat_template: Optional[str]) -> Optional[str]:
    if chat_template is None:
        return None
    try:
        with open(chat_template, "r") as f:
            resolved_chat_template = f.read()
    except OSError as e:
        JINJA_CHARS = "{}\n"
        if not any(c in chat_template for c in JINJA_CHARS):
            msg = (f"The supplied chat template ({chat_template}) "
                   f"looks like a file path, but it failed to be "
                   f"opened. Reason: {e}")
            raise ValueError(msg) from e

        # If opening a file fails, set chat template to be args to
        # ensure we decode so our escape are interpreted correctly
        resolved_chat_template = codecs.decode(chat_template, "unicode_escape")

    logger.info("Using supplied chat template:\n%s", resolved_chat_template)
    return resolved_chat_template


@lru_cache(maxsize=None)
def _image_token_str(model_config: ModelConfig,
                     tokenizer: PreTrainedTokenizer) -> Optional[str]:
    # TODO: Let user specify how to insert image tokens into prompt
    # (similar to chat template)
    model_type = model_config.hf_config.model_type
    if model_type == "phi3_v":
        # Workaround since this token is not defined in the tokenizer
        return "<|image_1|>"
    if model_type == "minicpmv":
        return "(<image>./</image>)"
    if model_type in ("blip-2", "chatglm", "fuyu", "paligemma"):
        # These models do not use image tokens in the prompt
        return None
    if model_type.startswith("llava"):
        return tokenizer.decode(model_config.hf_config.image_token_index)
    if model_type in ("chameleon", "internvl_chat"):
        return "<image>"
    raise TypeError(f"Unknown model type: {model_type}")


def _get_full_image_text_prompt(image_token_str: str, text_prompt: str) -> str:
    """Combine image and text prompts for vision language model"""

    # NOTE: For now we assume all model architectures use the same
    # image + text prompt format. This may change in the future.
    return f"{image_token_str}\n{text_prompt}"


def __dataframe_info_simple(table: Table) -> str:
    df_info_template_simple = (
        "/*\nDetails about the '{df_name}' "
        "dataframe that can be used as follows:\n{desc_info}\n*/")

    desc_info_lines = []
    for col in table["cols"]:

        comment_str = "means "+ col["comment"]  if col.get("comment") else ""
        data_type = col.get("dtype")

        contains_nan_str = "contains NaN, " if col.get("contains_nan",False) else ""

        unique_str = "is unique, " if col.get("unique",False) else ""

        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        exp_val = (VLLM_TABLE_INSERT_SEP_TOKEN + VLLM_TABLE_INSERT_EMBS_TOKEN +
                   VLLM_TABLE_INSERT_SEP_TOKEN)
        key = col.get("name")
        dil = (f"- '{key}' {data_type}, "
               f"{unique_str}{contains_nan_str}{comment_str}"
               f"Example Values: {exp_val}")

        desc_info_lines.append(dil)

    desc_info = "\n".join(desc_info_lines)

    desc_info = desc_info.replace(", '...']", ", ...]")

    df_info = df_info_template_simple.format(
        df_name=table["varname"],
        desc_info=desc_info,
    )

    return df_info


def __build_table_question(tables: List[Table]):
    pref = "## Details about the dataframes:\n\n"

    df_info_list = [
        __dataframe_info_simple(table) for table in tables
    ]
    return pref + '\n\n'.join(df_info_list)


def _get_full_table_text_prompt(tables: List[Table]) -> str:
    return __build_table_question(tables)


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    model_config: ModelConfig,
    tokenizer: PreTrainedTokenizer,
) -> ChatMessageParseResult:
    texts: List[str] = []
    mm_futures: List[Union[Awaitable[MultiModalDataDict],
                           MultiModalDataDict]] = []

    is_table = False
    for part in parts:
        part_type = part["type"]
        if part_type == "text":
            text = cast(ChatCompletionContentPartTextParam, part)["text"]
            texts.append(text)
        elif part_type == "image_url":
            if len(mm_futures) > 0:
                raise NotImplementedError(
                    "Multiple 'image_url' input is currently not supported.")

            image_url = cast(ChatCompletionContentPartImageParam,
                             part)["image_url"]

            if image_url.get("detail", "auto") != "auto":
                logger.warning(
                    "'image_url.detail' is currently not supported and "
                    "will be ignored.")

            image_future = async_get_and_parse_image(image_url["url"])
            mm_futures.append(image_future)

        elif part_type == "table":
            if len(mm_futures) > 0:
                raise NotImplementedError(
                    "Multiple 'table' input is currently not supported.")

            table_data:List[Table] = cast(ChatCompletionContentPartTableParam,
                             part)["table"]
            mm_futures.append({"table":table_data})
            is_table = True
        else:
            raise NotImplementedError(f"Unknown part type: {part_type}")

    text_prompt = "\n".join(texts)

    if mm_futures:
        if is_table:
            mm_data: MultiModalDataDict = mm_futures[0]
            text_prompt = _get_full_table_text_prompt(tables=mm_data["table"])

        else:
            image_token_str = _image_token_str(model_config, tokenizer)
            if image_token_str is not None:
                if image_token_str in text_prompt:
                    logger.warning(
                        "Detected image token string in the text prompt. "
                        "Skipping prompt formatting.")
                else:
                    text_prompt = _get_full_image_text_prompt(
                        image_token_str=image_token_str,
                        text_prompt=text_prompt,
                    )

    messages = [ConversationMessage(role=role, content=text_prompt)]

    return ChatMessageParseResult(messages=messages, mm_futures=mm_futures)


def _parse_chat_message_content(
    message: ChatCompletionMessageParam,
    model_config: ModelConfig,
    tokenizer: PreTrainedTokenizer,
) -> ChatMessageParseResult:
    role = message["role"]
    content = message.get("content")

    if content is None:
        return ChatMessageParseResult(messages=[], mm_futures=[])
    if isinstance(content, str):
        messages = [ConversationMessage(role=role, content=content)]
        return ChatMessageParseResult(messages=messages, mm_futures=[])

    return _parse_chat_message_content_parts(role, content, model_config,
                                             tokenizer)


def parse_chat_messages(
    messages: List[ChatCompletionMessageParam],
    model_config: ModelConfig,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[List[ConversationMessage], List[Awaitable[MultiModalDataDict]]]:
    conversation: List[ConversationMessage] = []
    mm_futures: List[Awaitable[MultiModalDataDict]] = []

    for msg in messages:
        parse_result = _parse_chat_message_content(msg, model_config,
                                                   tokenizer)

        conversation.extend(parse_result.messages)
        mm_futures.extend(parse_result.mm_futures)

    return conversation, mm_futures
