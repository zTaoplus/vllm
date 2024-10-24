import re
import random
from typing import List, Union
from functools import partial

from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.config import ModelConfig
from transformers import PreTrainedTokenizer

from .base import (
    MultiModalInputs,
    MultiModalPlugin,
    ColumnsTable,
    MarkdownTable,
)


logger = init_logger(__name__)


def table_custom_tokenize_encode(
    prompt: str, tokenizer: PreTrainedTokenizer, model_config: ModelConfig
) -> List[int]:
    """
    Tokenizes the input prompt by inserting a separator token
    between each chunk of text.

    Args:
        prompt (str): The input prompt to be tokenized.
                    It contains one or more instances of the
                    INSERT_EMBS_TOKEN.
        tokenizer (transformers.PreTrainedTokenizer):
            The tokenizer object used for tokenization.

    Returns:
       List[int]: The tokenized input prompt as a list of input IDs.

    """

    hf_config = model_config.hf_config

    # get placeholder is None then get the insert embeds token,
    # if not has the token , should raise
    placeholder_token = getattr(
        hf_config, "placeholder_token", None
    ) or getattr(hf_config.encoder_config, "insert_embs_token", None)
    placeholder_token_id = getattr(
        hf_config, "placeholder_token_id", None
    ) or getattr(hf_config.encoder_config, "insert_embs_token_id", None)

    if placeholder_token is None and placeholder_token_id is None:
        raise ValueError(
            "Cannot get place token and place token ids from model hf_config"
        )

    _repeated_count = 1 if hf_config.model_type.endswith("markup") else 3

    prompt_chunks = [
        tokenizer(
            e,
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        for e in prompt.split(placeholder_token)
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][
            :-1
        ]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(
        prompt_chunks, [placeholder_token_id] * _repeated_count * (offset + 1)
    ):
        input_ids.extend(x[offset:])
    return input_ids


def __dataframe_info_simple(
    table: ColumnsTable | MarkdownTable, model_config: ModelConfig
) -> str:
    if isinstance(table, MarkdownTable):
        return model_config.hf_config.placeholder_token + "\n"

    max_example_values = 3

    insert_embs_token = model_config.hf_config.encoder_config.insert_embs_token
    insert_seq_token = model_config.hf_config.encoder_config.insert_seq_token

    max_cols = model_config.hf_config.encoder_config.max_cols

    placeholder_val = insert_seq_token + insert_embs_token + insert_seq_token

    desc_info_lines = []
    tables_sampled = table["columns"]
    # TODO: TO BE HOEST, THE COLUMNS is the actually needed infos for the table
    # the value not more important than columns
    # so do not sample cols, if cols* rows gt 500？ we consider sample the rows
    # and also we should ensure the sampled rows  > 3
    if max_cols < len(table["columns"]):
        print(f"""The model supports a maximum of {max_cols} columns, 
but {len(table['columns'])} were received. 
Currently, {max_cols} records have been sampled to continue the process. 
If the results are unsatisfactory, 
please ensure that the length of the columns in the table is ≤ {max_cols}""")

        tables_sampled = random.sample(table["columns"], max_cols)

    for col in tables_sampled:
        n = col["name"]
        tp = col["dtype"] + ","
        isu = ""
        if col.get("is_unique"):
            isu = "is unique,"

        if isu and ("float" in tp or "int" in tp):
            isu = ""

        unique_col_vals = set(col["values"])

        ctn = "contains NaN," if col.get("contains_nan") else ""
        expval_prefx = "Example Values: "
        expval = str(
            random.sample(
                list(unique_col_vals),
                min(len(unique_col_vals), max_example_values),
            )
        )

        if len(unique_col_vals) > max_example_values:
            expval = expval.rsplit("]", maxsplit=1)[0] + ", ...]"

        desc_info_lines.append(
            f"{placeholder_val} '{n}' {tp}{isu}{ctn}{expval_prefx+expval}"
        )

    desc_info = "\n".join(desc_info_lines)

    return f"{desc_info}"


def __build_table_question(
    tables: Union[List[ColumnsTable] | List[MarkdownTable]],
    model_config: ModelConfig,
):
    f = partial(__dataframe_info_simple, model_config=model_config)

    df_info_list = [f(table) for table in tables]
    return df_info_list


def get_full_tables_info(
    tables: Union[List[ColumnsTable] | List[MarkdownTable]],
    model_config: ModelConfig,
    return_text=True,
) -> str:
    if not isinstance(tables, list):
        tables = [tables]
    return (
        "".join(__build_table_question(tables, model_config))
        if return_text
        else __build_table_question(tables, model_config)
    )


def get_full_tarbular_text_prompt(
    placeholder_token_str: str, table_infos: List[str], text_prompt: str
) -> str:
    occurrences = re.findall(re.escape(placeholder_token_str), text_prompt)
    if len(occurrences) == 0:
        return f"{placeholder_token_str}\n{text_prompt}"

    if len(occurrences) != len(table_infos):
        raise ValueError(
            f"The length of table_infos::List"
            f"must match the number of occurrences of "
            f"`{placeholder_token_str}`."
        )

    return re.sub(
        re.escape(placeholder_token_str),
        lambda match: table_infos.pop(0),
        text_prompt,
    )


class TablePlugin(MultiModalPlugin):
    """Plugin for table data."""

    def get_data_key(self) -> str:
        return "table"

    def _default_input_mapper(
        self, ctx: InputContext, data: object
    ) -> MultiModalInputs:
        raise NotImplementedError("There is no default table input mapper")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        raise NotImplementedError(
            "There is no default maximum multimodal tokens"
        )
