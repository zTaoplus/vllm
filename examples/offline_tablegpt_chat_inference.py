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
    max_num_seqs=2,
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


# results = model.chat(messages=batch_msgs, sampling_params=p)
# print(results)


empty_values_msgs = [
    {"role": "system", "content": "You are a helpul assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """With access to several pandas dataframes, your task is to write Python code to address the user's question.

## Use the format below:
Question: The user's question.
Thought: Consider the question and the provided dataframes to determine the appropriate approach.
Python code: Provide the Python code, wrapped in ```python ... ```.

## Details about each dataframe:""",
            },
            {
                "type": "text",
                "text": f"/* Details about the df_{1} other info as follows:\n",
            },
            {
                "type": "table",
                "table": {
                    "columns": [
                        {
                            "name": "district_id",
                            "dtype": "INTEGER",
                            "values": [],
                            "contains_nan": False,
                            "is_unique": False,
                        },
                        {
                            "name": "average_salary",
                            "dtype": "REAL",
                            "values": [],
                            "contains_nan": False,
                            "is_unique": False,
                        },
                    ]
                },
            },
            {
                "type": "text",
                "text": "*/",
            },
            {
                "type": "text",
                "text": "Question: 帮我查看数据前10行",
            },
        ],
    },
]

a = [
    {
        "role": "system",
        "content": "You are 数海闻涛, an expert Python data analyst developed by 浙江大学计算机创新技术研究院 (Institute of Computer Innovation of Zhejiang University, or ZJUICI). Your job is to help user analyze datasets by writing Python code. Each markdown codeblock you write will be executed in an IPython environment, and you will receive the execution output. You should provide results analysis based on the execution output.\nFor politically sensitive questions, security and privacy issues, or other non-data analyze questions, you will refuse to answer.\n\nRemember:\n- Comprehend the user's requirements carefully & to the letter.\n- If additional information is needed, feel free to ask the user.\n- Give a brief description for what you plan to do & write Python code.\n- You can use `read_df(uri: str) -> pd.DataFrame` function to read different file formats into DataFrame.\n- When creating charts, prefer using `seaborn`.\n- If error occurred, try to fix it.\n- Response in the same language as the user.\n- Today is 2024-09-26",
    },
    {"role": "user", "content": "文件名称: 'patients.csv'"},
    {
        "role": "assistant",
        "content": "我已经收到您的数据文件，我需要查看文件内容以对数据集有一个初步的了解。首先我会读取数据到 `df1` 变量中，并通过 `df1.info` 查看 NaN 情况和数据类型。\n\n```python\n# Load the data into a DataFrame\ndf1 = read_df('patients.csv')\n\n# Remove leading and trailing whitespaces in column names\ndf1.columns = df1.columns.str.strip()\n\n# Remove rows and columns that contain only empty values\ndf1 = df1.dropna(how='all').dropna(axis=1, how='all')\n\n# Get the basic information of the dataset\ndf1.info(memory_usage=False)\n```",
    },
    {
        "role": "system",
        "content": "```pycon\n<class 'pandas.core.frame.DataFrame'>\nRangeIndex: 2 entries, 0 to 1\nData columns (total 4 columns):\n #   Column      Non-Null Count  Dtype \n---  ------      --------------  ----- \n 0   patient_id  2 non-null      int64 \n 1   age         2 non-null      int64 \n 2   gender      2 non-null      object\n 3   state       2 non-null      object\ndtypes: int64(2), object(2)\n```",
    },
    {
        "role": "assistant",
        "content": "接下来我将用 `df1.head(3)` 来查看数据集的前 3 行。\n\n```python\n# Show the first 3 rows to understand the structure\ndf1.head(3)\n```",
    },
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "```pycon\n   patient_id  age  gender       state\n0           1   35  Female  California\n1           2   45    Male       Texas\n```",
            },
            {
                "type": "text",
                "text": "/*\nDetails about the 'df1' other info as follows:\n",
            },
            {
                "type": "table",
                "table": {
                    "columns": [
                        {
                            "name": "patient_id",
                            "dtype": "int64",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": [1, 2],
                        },
                        {
                            "name": "age",
                            "dtype": "int64",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": [35, 45],
                        },
                        {
                            "name": "gender",
                            "dtype": "object",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": ["Female", "Male"],
                        },
                        {
                            "name": "state",
                            "dtype": "object",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": ["California", "Texas"],
                        },
                    ]
                },
            },
            {"type": "text", "text": "*/"},
        ],
    },
    {
        "role": "assistant",
        "content": "我已经了解了数据集 patients.csv 的基本信息。请问我可以帮您做些什么？",
    },
    {"role": "user", "content": "文件名称: 'treatments.csv'"},
    {
        "role": "assistant",
        "content": "我已经收到您的数据文件，我需要查看文件内容以对数据集有一个初步的了解。首先我会读取数据到 `df2` 变量中，并通过 `df2.info` 查看 NaN 情况和数据类型。\n\n```python\n# Load the data into a DataFrame\ndf2 = read_df('treatments.csv')\n\n# Remove leading and trailing whitespaces in column names\ndf2.columns = df2.columns.str.strip()\n\n# Remove rows and columns that contain only empty values\ndf2 = df2.dropna(how='all').dropna(axis=1, how='all')\n\n# Get the basic information of the dataset\ndf2.info(memory_usage=False)\n```",
    },
    {
        "role": "system",
        "content": "```pycon\n<class 'pandas.core.frame.DataFrame'>\nRangeIndex: 2 entries, 0 to 1\nData columns (total 5 columns):\n #   Column           Non-Null Count  Dtype \n---  ------           --------------  ----- \n 0   treatment_id     2 non-null      int64 \n 1   patient_id       2 non-null      int64 \n 2   treatment        2 non-null      object\n 3   date             2 non-null      object\n 4   completion_date  1 non-null      object\ndtypes: int64(2), object(3)\n```",
    },
    {
        "role": "assistant",
        "content": "接下来我将用 `df2.head(3)` 来查看数据集的前 3 行。\n\n```python\n# Show the first 3 rows to understand the structure\ndf2.head(3)\n```",
    },
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "```pycon\n   treatment_id  patient_id   treatment        date completion_date\n0             1           1        MBSR  2021-01-01      2021-03-01\n1             2           2  Medication  2021-01-02             NaN\n```",
            },
            {
                "type": "text",
                "text": "/*\nDetails about the 'df2' other info as follows:\n",
            },
            {
                "type": "table",
                "table": {
                    "columns": [
                        {
                            "name": "treatment_id",
                            "dtype": "int64",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": [1, 2],
                        },
                        {
                            "name": "patient_id",
                            "dtype": "int64",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": [1, 2],
                        },
                        {
                            "name": "treatment",
                            "dtype": "object",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": ["MBSR", "Medication"],
                        },
                        {
                            "name": "date",
                            "dtype": "object",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": ["2021-01-01", "2021-01-02"],
                        },
                        {
                            "name": "completion_date",
                            "dtype": "object",
                            "contains_nan": True,
                            "is_unique": False,
                            "values": ["2021-03-01", "nan"],
                        },
                    ]
                },
            },
            {
                "type": "table",
                "table": {
                    "columns": [
                        {
                            "name": "treatment_id",
                            "dtype": "int64",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": [1, 2],
                        },
                        {
                            "name": "patient_id",
                            "dtype": "int64",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": [1, 2],
                        },
                        {
                            "name": "treatment",
                            "dtype": "object",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": ["MBSR", "Medication"],
                        },
                        {
                            "name": "date",
                            "dtype": "object",
                            "contains_nan": False,
                            "is_unique": True,
                            "values": ["2021-01-01", "2021-01-02"],
                        },
                        {
                            "name": "completion_date",
                            "dtype": "object",
                            "contains_nan": True,
                            "is_unique": False,
                            "values": ["2021-03-01", "nan"],
                        },
                    ]
                },
            },
            {"type": "text", "text": "*/"},
        ],
    },
    {
        "role": "assistant",
        "content": "我已经了解了数据集 treatments.csv 的基本信息。请问我可以帮您做些什么？",
    },
    {
        "role": "user",
        "content": "我们需要对所有接受了'MBSR'治疗方式的患者的治疗完成日期进行更新，将其设置为'2023-12-31'。请先查询出这些患者的治疗编号和当前的治疗完成日期，然后进行数据更新。需要联合患者信息和治疗信息表，通过患者编号关联。",
    },
]


# model.llm_engine.model_config.hf_config.encoder_config.max_cols = 2
# model.llm_engine.model_config.multimodal_config.limit_per_prompt = {"table":3}
batch_msgs.append(empty_values_msgs)
batch_msgs.append(a)
results = model.chat(
    messages=batch_msgs, sampling_params=p
)
# results = model.chat(messages=a, sampling_params=p)

print(results)
