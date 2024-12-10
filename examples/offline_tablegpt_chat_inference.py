# 1. git clone https://github.com/zTaoplus/vllm.git
# install tablegpt vllm

## build from source (dev recommended)
## Note: Building from source may take 10-30 minutes and requires access to
# GitHub or other repositories. Make sure to configure an HTTP/HTTPS proxy.
## cd vllm && pip install -e . [-v]. The -v flag is optional
# and can be used to display verbose logs.

# see https://github.com/zTaoplus/TableGPT-hf to view the model-related configs.

from io import StringIO

import pandas as pd

from vllm import LLM
from vllm.sampling_params import SamplingParams

DEFAULT_SYS_MSG = "You are a helpful assistant."
ENCODER_TYPE = "contrastive"

# TODO: should migrate to hf model repo name
# model_name_or_path = "/zt/encoder/encoded_model"

# markup model
model_name_or_path = "/zt/encoder/tablegpt-t5/merged/test-hf-load"
ENCODER_TYPE = "markup"

model = LLM(
    model=model_name_or_path,
    max_model_len=12580,
    # max_num_seqs=10,
    # max_num_batched_tokens =  125800,
    dtype="bfloat16",
    limit_mm_per_prompt={"table": 2},
)

p = SamplingParams(temperature=0, max_tokens=2048)


def extract_df_info(df: pd.DataFrame):
    sio = StringIO()
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.info(buf=sio, memory_usage=False)

    return sio.getvalue()


def extract_contrastive_table(df: pd.DataFrame):
    return {
        "columns": [{
            "name": col,
            "dtype": str(df[col].dtype),
            "contains_nan": df[col].isnull().any(),
            "is_unique": df[col].nunique() == len(df[col]),
            "values": df[col].tolist(),
        } for col in df.columns]
    }


def extract_markup_table(df: pd.DataFrame):
    return df.head(200).to_markdown()


def print_promt(res):
    print("------------------PROMPT Start----------------")
    print(res.prompt)
    print("------------------PROMPT END-----------------")

    print("++++++++++++++++++++++++Response Start++++++++++++++++++++++++")
    print(res.outputs[0].text)
    print("++++++++++++++++++++++++Response End++++++++++++++++++++++++")


batch_msgs = []

a_tableqa_msg_with_empty_values = [
    {
        "role": "system",
        "content": "You are a helpul assistant."
    },
    {
        "role":
        "user",
        "content": [
            {
                "type":
                "text",
                "text":
                f"/* Details about the df_{1} \
other info as follows:\n <TABLE_CONTENT>\n*/",
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
        ],
    },
]

multi_tables_in_multi_turns_msg = [
    {
        "role": "system",
        "content": "You are a helpul assistant.",
    },
    {
        "role":
        "system",
        "content": [
            {
                "type":
                "text",
                "text":
                "/*\nDetails about the 'df1' \
other info as follows:\n<TABLE_CONTENT>*/\n",
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
        ],
    },
    {
        "role": "assistant",
        "content": "我已经了解了数据集 patients.csv 的基本信息。\
请问我可以帮您做些什么？",
    },
    {
        "role": "user",
        "content": "文件名称: 'treatments.csv'"
    },
    {
        "role":
        "system",
        "content": [
            {
                "type":
                "text",
                "text":
                "/*\nDetails about the \
'df2' other info as follows:\n<TABLE_CONTENT>\n*/",
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
        ],
    },
    {
        "role": "assistant",
        "content": "我已经了解了数据集 treatments.csv 的基本信息。\
请问我可以帮您做些什么？",
    },
    {
        "role":
        "user",
        "content":
        "我们需要对所有接受了'MBSR'治疗方式的患者\
的治疗完成日期进行更新,\
将其设置为'2023-12-31'。请先查询出这些患者的治疗编号和当前的治疗完成日期，\
然后进行数据更新。需要联合患者信息和治疗信息表，通过患者编号关联。",
    },
]

# batch_msgs.append(a_tableqa_msg_with_empty_values)
batch_msgs.append(multi_tables_in_multi_turns_msg)

results = model.chat(messages=batch_msgs, sampling_params=p)
for res in results:
    print("=" * 10 + "Prompt Start" + "=" * 10)
    print(res.prompt)
    print("=" * 10 + "Prompt End" + "=" * 10)

    print("=" * 10 + "Response Start" + "=" * 10)
    print(res.outputs[0].text)
    print("=" * 10 + "Response End" + "=" * 10)
