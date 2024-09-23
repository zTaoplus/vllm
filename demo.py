import pandas as pd


import torch

from transformers import AutoModel,AutoTokenizer

model_path = "/zt/encoder/encoded_model/test-hf-load"

# model_markup_path = "/zt/encoder/tablegpt-t5/merged/test-hf-load"

# max_rows = 30

table_1 = pd.read_csv(
   "/root/workspace/vllm/spotify_tracks.csv",
    nrows=30,
    encoding="utf-8"
)

# table_2 = pd.read_csv(
#     "/root/workspace/vllm/TV_show_data_versionskz.csv",
#     nrows=500,
#     encoding="utf-8"
# )


# tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path,trust_remote_code=True,torch_dtype=torch.bfloat16).eval().cuda()


# # 单表单轮对话
# query = "帮我筛选时长最长的3首歌曲"

# resp, history = model.chat(
#     tokenizer,
#     query,
#     # table_1,
#     {
#         "table_1":table_1,
#         # "table_2":table_2
#     },
#     return_history=True,
#     max_new_tokens=1024,
#     verbose=True,
#     # merge_embeds_type="original"
#     merge_embeds_type="vllm"
# )

# print(f"Question: \n{query}\n-----\nAssistant: \n{resp}\n------\nHistory:\n{history}")


# query2 = "在所有“Science-Fiction”类型的剧集中，哪些剧集的播放时间在晚上8点以后，并且评分低于6.5？"
# # 多表多轮对话
# resp, history = model.chat(
#     tokenizer,
#     query2,
#     # {
#     #     "table_1":table_1,
#     #     "table_2":table_2
#     # },
#     history=history,
#     return_history=True,
#     max_new_tokens=1024,
#     verbose=True
# )
# print(f"Question: \n{query2}\n------\nAssistant: \n{resp}\n------\nHistory:\n{history}")

# resp, history = model.chat(
#     tokenizer,
#     query,
#     [table_1,table_2],
#     verbose=True
# )

# print(f"Question: {query}, Assistant: {resp}, history:{history}")


# # 多表多轮对话
# query = "帮我筛选时长最长的3首歌曲"



# --------------- markup demo ===================
model_markup_path = "/zt/encoder/tablegpt-t5/merged/test-hf-load"

tokenizer = AutoTokenizer.from_pretrained(model_markup_path,trust_remote_code=True)
model = AutoModel.from_pretrained(model_markup_path,trust_remote_code=True,torch_dtype=torch.bfloat16).eval().cuda()


# 单表单轮对话
query = "帮我筛选时长最长的3首歌曲"

resp, history = model.chat(
    tokenizer,
    query,
    # table_1,
    {
        "table_1":table_1,
        # "table_2":table_2
    },
    return_history=True,
    max_new_tokens=1024,
    verbose=True,
    # merge_embeds_type="original"
    # merge_embeds_type="vllm"
)

print(f"Question: \n{query}\n-----\nAssistant: \n{resp}\n------\nHistory:\n{history}")
