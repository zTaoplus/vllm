# 1. git clone -b v0.6.3-tbe https://github.com/zTaoplus/vllm.git
# install tablegpt vllm


##  apply diff file (recommended in case of use only)
# 1. pip install vllm==v0.6.3
# 2. git diff fd47e57f4b0d5f7920903490bce13bc9e49d8dba HEAD -- vllm | patch -p1 -d "$(pip show vllm | grep Location | awk '{print $2}')"

## build from source (dev recommended)
## Note: Building from source may take 10-30 minutes and requires access to GitHub or other repositories. Make sure to configure an HTTP/HTTPS proxy.
## cd vllm && pip install -e . [-v]. The -v flag is optional and can be used to display verbose logs.


# see https://github.com/zTaoplus/TableGPT-hf to view the model-related configs.

from vllm import LLM
from vllm.sampling_params import SamplingParams


import pandas as pd

from typing import Literal, List, Optional

from io import StringIO

DEFAULT_SYS_MSG = "You are a helpful assistant."
ENCODER_TYPE = "contrastive"

model_name_or_path = "/zt/encoder/encoded_model"

# markup model
# model_name_or_path = "/zt/encoder/tablegpt-t5/merged/test-hf-load"
# ENCODER_TYPE = "markup"

model = LLM(
    model=model_name_or_path,
    max_model_len=8192,
    max_num_seqs=16,
    dtype="bfloat16",
    limit_mm_per_prompt={"table": 3},
)

p = SamplingParams(temperature=0, max_tokens=2048)


def extract_df_info(df: pd.DataFrame):
    sio = StringIO()
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.info(buf=sio, memory_usage=False)

    return sio.getvalue()


def extract_contrastive_table(df: pd.DataFrame):
    # Convert DataFrame to the desired format
    # contains nan
    # is unique
    return {
        "columns": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "contains_nan": df[col].isnull().any(),
                "is_unique": df[col].nunique() == len(df[col]),
                "values": df[col].tolist(),  # slice?
            }
            for col in df.columns
        ]
    }


def extract_markup_table(df: pd.DataFrame):
    return df.head(200).to_markdown()


def build_messages_from_csv_file(
    csv_paths: List[str],
    query: str,
    system_message: Optional[str] = None,
    encoder_type: Literal["contrastive", "markup"] = "contrastive",
):
    messages = [
        {
            "role": "system",
            "content": system_message if system_message else DEFAULT_SYS_MSG,
        }
    ]

    for idx, csv_path in enumerate(csv_paths, 1):
        df = pd.read_csv(csv_path, encoding="utf-8", nrows=500)

        messages.extend(
            [
                {"role": "user", "content": f"file path: {csv_path}"},
                {
                    "role": "ai",
                    "content": f"我已经收到您的数据文件，我需要查看文件内容以对数据集有一个初步的了解。首先我会读取数据到 df_{idx} 变量中，并通过 df_{idx}.info 查看 NaN 情况和数据类型。\n```python\n# Load the data into a DataFrame\ndf_{idx} = read_df('{csv_path}')\n\n# Remove leading and trailing whitespaces in column names\ndf_{idx}.columns = df_{idx}.columns.str.strip()\n\n# Remove rows and columns that contain only empty values\ndf_{idx} = df_{idx}.dropna(how='all').dropna(axis=1, how='all')\n\n# Get the basic information of the dataset\ndf_{idx}.info(memory_usage=False)\n```\n",
                },
                {"role": "system", "content": extract_df_info(df)},
                {
                    "role": "ai",
                    "content": f"接下来我将用 `df_{idx}.head(5)` 来查看数据集的前 5 行。\n```python\n# Show the first 5 rows to understand the structure\ndf_{idx}.head(5)```\n",
                },
                {
                    "role": "system",
                    "content": [
                        # {"type": "text", "text": str(df.head(3))},
                        {"type": "text", "text": "df1 的信息如下:\n"},
                        {
                            "type": "table",
                            "table": extract_contrastive_table(df)
                            if encoder_type == "contrastive"
                            else extract_markup_table(df),
                        },
                        # {"type": "text", "text": "df2 的信息如下:\n"},
                        # {
                        #     "type": "table",
                        #     "table": extract_contrastive_table(df)
                        #     if encoder_type == "contrastive"
                        #     else extract_markup_table(df),
                        # },
                    ],
                },
                {
                    "role": "ai",
                    "content": f"我已经了解了数据集 {csv_path} 的基本信息。请问我可以帮您做些什么？",
                },
            ]
        )

    messages.append({"role": "user", "content": query})

    return messages


def build_tableqa_messages_from_csv_file(
    csv_paths: List[str],
    query: str,
    system_message: Optional[str] = None,
    encoder_type: Literal["contrastive", "markup"] = "contrastive",
):
    messages = [
        {
            "role": "system",
            "content": system_message if system_message else DEFAULT_SYS_MSG,
        }
    ]

    tabular_part = [
        {
            "type": "text",
            "text": """With access to several pandas dataframes, your task is to write Python code to address the user's question.

## Use the format below:
Question: The user's question.
Thought: Consider the question and the provided dataframes to determine the appropriate approach.
Python code: Provide the Python code, wrapped in ```python ... ```.

## Details about each dataframe:""",
        }
    ]

    for idx, csv_path in enumerate(csv_paths, 1):
        df = pd.read_csv(csv_path, encoding="utf-8", nrows=500)
        tabular_part.append(
            {
                "type": "text",
                "text": f"/* Details about the df_{idx} other info as follows:\n",
            }
        )
        tabular_part.append(
            {
                "type": "table",
                "table": extract_contrastive_table(df)
                if encoder_type == "contrastive"
                else extract_markup_table(df),
            }
        )
        tabular_part.append({"type": "text", "text": "*/"})

    tabular_part.append({"type": "text", "text": f"Questions: {query}"})

    messages.append({"role": "user", "content": tabular_part})

    # messages.append({"role": "user", "content": query})

    return messages


def print_promt(res):
    print("------------------PROMPT Start----------------")
    print(res.prompt)
    print("------------------PROMPT END-----------------")

    print("++++++++++++++++++++++++Response Start++++++++++++++++++++++++")
    print(res.outputs[0].text)
    print("++++++++++++++++++++++++Response End++++++++++++++++++++++++")


batch_msgs = []
for csv_path, user_query in [
    (
        (
            "/root/workspace/vllm/spotify_tracks.csv",
            "/root/workspace/vllm/TV_show_data_versionskz.csv",
        ),
        "查看时长最长的三首歌曲",
    ),
    (
        "/root/workspace/vllm/TV_show_data_versionskz.csv",
        "在所有“Science-Fiction”类型的剧集中，哪些剧集的播放时间在晚上8点以后，并且评分超过8.5？",
    ),
]:
    if isinstance(csv_path, (list, tuple)):
        messages = build_tableqa_messages_from_csv_file(
            csv_path, user_query, encoder_type=ENCODER_TYPE
        )
        batch_msgs.append(messages)
    elif isinstance(csv_path, str):
        messages = build_tableqa_messages_from_csv_file(
            [csv_path], user_query, encoder_type=ENCODER_TYPE
        )

        batch_msgs.append(messages)
    else:
        raise ValueError(
            f"Unexpected csv path, expect `List[str] | Tuple[str] | str` but given {type(csv_path)}"
        )


results = model.chat(messages=batch_msgs, sampling_params=p)

for res in results:
    print_promt(res)
