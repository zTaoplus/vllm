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
from vllm.multimodal.utils import (async_get_and_parse_image,
                                   get_and_parse_table)

logger = init_logger(__name__)


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


class TableURL(TypedDict, total=False):
    url: Required[List[str]]
    """Either a URL of the image or the base64 encoded image data."""


class ChatCompletionContentPartTableParam(TypedDict, total=False):
    table_url: Required[TableURL]

    type: Required[Literal["table_url"]]
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


def __dataframe_info_simple(df: pd.DataFrame, df_name: str, comments=None):
    df_info_template_simple = (
        "/*\nDetails about the '{df_name}' "
        "dataframe that can be used as follows:\n{desc_info}\n*/")
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    if comments is not None:
        comments_dict = {
            item["content"]: {
                "comment": item["comment"],
                "info": item["info"]
            }
            for item in comments
        }
        comment_value = info_df['Column Name'].apply(
            lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
    desc_info_dict = info_df_new.to_dict()

    desc_info_lines = []
    for key, value in desc_info_dict.items():
        comment_str = "means " + value['Comment'] + "." if value.get(
            "Comment", "") else ""
        data_type = value["Data Type"]

        contains_nan_str = "contains NaN, " if value["Contains NaN"] else ""

        unique_str = "is unique, " if value["Is Unique"] else ""

        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        exp_val = (VLLM_TABLE_INSERT_SEP_TOKEN + VLLM_TABLE_INSERT_EMBS_TOKEN +
                   VLLM_TABLE_INSERT_SEP_TOKEN)
        dil = (f"- '{key}' {data_type}, "
               f"{unique_str}{contains_nan_str}{comment_str}"
               f"Example Values: {exp_val}")
        desc_info_lines.append(dil)

    desc_info = "\n".join(desc_info_lines)

    desc_info = desc_info.replace(", '...']", ", ...]")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )

    return df_info


def __build_table_question(dfs: List[pd.DataFrame], query: str, comments=None):
    """
    Build the instruction text for the user question.

    Args:
        conv (dict): A dictionary containing conversation information. 
                     It should contain the following keys: 
                        csv_abs_paths, df_names, query.

    Returns:
        str: The generated question string.

    """
    pref = (
        "With several pandas dataframes available, "
        "your task is to write the Python code to address the user's question."
        "\n\n## Follow this format:"
        "\nQuestion: The user's query.\n"
        "Thought: "
        "Evaluate the dataframes and the question to determine the solution."
        "\nPython code: Generate the Python code, within ```python ... ```."
        "\n\n## Details about the dataframes:\n\n")

    df_info_list = [
        __dataframe_info_simple(df, df_name, comments) for df_name, df in dfs
    ]
    suf = '''\n\nQuestion: ''' + query + '\n'
    return pref + '\n\n'.join(df_info_list) + suf


def _get_full_table_text_prompt(dfs: List[pd.DataFrame],
                                text_prompt: str) -> str:

    # build a table question and add the table token strings
    return __build_table_question(dfs, text_prompt)


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

        elif part_type == "table_url":
            if len(mm_futures) > 0:
                raise NotImplementedError(
                    "Multiple 'table_url' input is currently not supported.")

            table_url = cast(ChatCompletionContentPartTableParam,
                             part)["table_url"]

            table_data = get_and_parse_table(table_url["url"])
            mm_futures.append(table_data)
            is_table = True
        else:
            raise NotImplementedError(f"Unknown part type: {part_type}")

    text_prompt = "\n".join(texts)

    if mm_futures:
        if is_table:
            mm_data: MultiModalDataDict = mm_futures[0]
            text_prompt = _get_full_table_text_prompt(dfs=mm_data["table"],
                                                      text_prompt=text_prompt)

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
