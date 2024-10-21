# 1. git clone -b v0.5.5-tablegpt-merged https://github.com/zTaoplus/vllm.git
# install tablegpt vllm


##  apply diff file (recommended in case of use only)
# 1. pip install vllm==0.5.5
# 2. cd vllm
# 3. git diff 09c7792610ada9f88bbf87d32b472dd44bf23cc2 HEAD -- vllm | patch -p1 -d "$(pip show vllm | grep Location | awk '{print $2}')"

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

model = LLM(
    model=model_name_or_path,
    max_model_len=8192,
    max_num_seqs=16,
    dtype="bfloat16",
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
    return {
        "columns": [
          {
              "name": col,
              "dtype": str(df[col].dtype),
              "contains_nan": df[col].isnull().any(),
              "is_unique":df[col].nunique() == len(df[col]),
              "values": df[col].tolist(),  # slice?
          }
          for col in df.columns
      ]
    }


def extract_markup_table(df: pd.DataFrame):
    return df.head(200).to_markdown()


def build_messages_multi_file_turn_from_csv_file(
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
                # {"role": "user", "content": f"file path: {csv_path}"},
                # {
                #     "role": "ai",
                #     "content": f"我已经收到您的数据文件，我需要查看文件内容以对数据集有一个初步的了解。首先我会读取数据到 df_{idx} 变量中，并通过 df_{idx}.info 查看 NaN 情况和数据类型。\n```python\n# Load the data into a DataFrame\ndf_{idx} = read_df('{csv_path}')\n\n# Remove leading and trailing whitespaces in column names\ndf_{idx}.columns = df_{idx}.columns.str.strip()\n\n# Remove rows and columns that contain only empty values\ndf_{idx} = df_{idx}.dropna(how='all').dropna(axis=1, how='all')\n\n# Get the basic information of the dataset\ndf_{idx}.info(memory_usage=False)\n```\n",
                # },
                # {"role": "system", "content": extract_df_info(df)},
                # {
                #     "role": "ai",
                #     "content": f"接下来我将用 `df_{idx}.head(5)` 来查看数据集的前 5 行。\n```python\n# Show the first 5 rows to understand the structure\ndf_{idx}.head(5)```\n",
                # },
                {
                    "role": "system",
                    "content": [
                        # {"type": "text", "text": df.head(5).to_string()},
                        {"type": "text", "text": "/* Detail of the df1 as follow:\n<TABLE_CONTENT>\n*/"},
                        {
                            "type": "table",
                            "tables": [extract_contrastive_table(df)]
                            if encoder_type == "contrastive"
                            else extract_markup_table(df),
                        }
                    ],
                    "tool_call_id":"tool"
                },
            #     {
            #         "role": "ai",
            #         "content": f"我已经了解了数据集 {csv_path} 的基本信息。请问我可以帮您做些什么？",
            #     },
            ]
        )

    messages.append({"role": "user", "content": query})

    return messages


def build_messages_one_file_turn_from_csv_file(
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

    tables = []
    for _, csv_path in enumerate(csv_paths, 1):
        df = pd.read_csv(csv_path, encoding="utf-8", nrows=500)

        tables.append(
            extract_contrastive_table(df)
            if encoder_type == "contrastive"
            else extract_markup_table(df)
        )

    messages.append(
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "".join(
                        f"df_{i} as follow:\n<TABLE_CONTENT>\n" 
                        for i in range(len(tables))),
                },
                {"type": "table", "tables": tables},
            ],
            "tool_call_id":"tool"
        }
    )

    messages.append({"role": "user", "content": query})

    return messages

def print_prompts(res):
    print("------------------PROMPT Start----------------")
    print(res[0].prompt)
    print("------------------PROMPT END-----------------")

    print("------------------PROMPT ID Start----------------")
    print(res[0].prompt_token_ids)
    print("------------------PROMPT ID END-----------------")

    print("++++++++++++++++++++++++Response Start++++++++++++++++++++++++")
    print(res[0].outputs[0].text)
    print("++++++++++++++++++++++++Response End++++++++++++++++++++++++")
  
# NOTE: You can use your custom dataset loader, but note that the vllm.LLM cannot load multiple different models onto different GPUs within the same machine.
result = []
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
        "查看播放最好的电视剧类型",
    ),
]:
    if isinstance(csv_path, (list, tuple)):
        messages = build_messages_one_file_turn_from_csv_file(
            csv_path, user_query, encoder_type=ENCODER_TYPE
        )
        res = model.chat(messages=messages, sampling_params=p)
        result.append(res)
        print_prompts(res)
        # messages = build_messages_multi_file_turn_from_csv_file(
        #     csv_path, user_query, encoder_type=ENCODER_TYPE
        # )

        # res = model.chat(messages=messages, sampling_params=p)
        # print_prompts(res)
        # result.append(res)

    elif isinstance(csv_path, str):
        messages = build_messages_multi_file_turn_from_csv_file(
            [csv_path], user_query, encoder_type=ENCODER_TYPE
        )
        res = model.chat(messages=messages, sampling_params=p)
        print_prompts(res)
        result.append(res)
    else:
        raise ValueError(
            f"Unexpected csv path, expect `List[str] | Tuple[str] | str` but given {type(csv_path)}"
        )

# print(result)
