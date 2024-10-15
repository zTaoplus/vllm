# 1. git clone -b v0.5.5-tablegpt-merged https://github.com/zTaoplus/vllm.git
# install tablegpt vllm


##  apply diff file (recommended in case of use only)
# 1. pip install vllm==0.5.5
# 2. cd vllm
# 3. git diff 09c7792610ada9f88bbf87d32b472dd44bf23cc2 HEAD -- vllm | patch -p1 -d "$(pip show vllm | grep Location | awk '{print $2}')"

## build from source (dev recommended)
## Note: Building from source may take 10-30 minutes and requires access to GitHub or other repositories. Make sure to configure an HTTP/HTTPS proxy.
## cd vllm && pip install -e . [-v]. The -v flag is optional and can be used to display verbose logs.


from vllm import LLM
from vllm.sampling_params import SamplingParams


# Currently, only a single GPU is supported for serving the model.
# For the LongLin (contrastive) encoder in TableGPT, you should set the maximum number of sequences to 16 or fewer when using the Nvidia A100 40G.

# Below is the file structure of the TableGPT model.

# longlin's Model (tablegpt_contrastive)
"""
├── config.json
├── configuration_tablegpt.py
├── configuration_tablegpt_enc.py
├── encoder  # copied from  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 , and removed model file
│   ├── 1_Pooling
│   │   └── config.json
│   ├── config.json
│   ├── config_sentence_transformers.json
│   ├── data_config.json
│   ├── modules.json
│   ├── sentence_bert_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── train_script.py
│   └── vocab.txt
├── generation_config.json
├── merges.txt
├── model.safetensors
├── modeling_tablegpt.py  # for hf load
├── modeling_tablegpt_encoder.py # for hf load
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── utils.py # for hf load
└── vocab.json
"""

# and config.json about the model
"""
{
  "architectures": [
    "TableGPTContrastiveForCausalLM"
  ],
  "auto_map":{
    "AutoConfig": "configuration_tablegpt.TableGPTConfig"
  },
  "model_type": "tablegpt_contrastive",
  "torch_dtype": "bfloat16",
  "is_encoder_decoder":false,
  "llm_config": {
    "_name_or_path": "Qwen/Qwen2-7B",
    "architectures": [
      "Qwen2ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 131072,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.41.2",
    "use_cache": false,
    "use_sliding_window": false,
    "vocab_size": 152064
  },
  "encoder_config": {
    "num_cols": 20,
    "depth": 12,
    "heads": 16,
    "attn_dropout": 0.1,
    "ff_dropout": 0.1,
    "cont_dim":256,
    "attentiontype": "colrow",
    "pred_type": "contrastive",
    "dim_head": 64,
    "pooling": "mean",
    "col_name": false,
    "numeric_mlp": false,
    "max_rows": 50,
    "max_cols": 100,
    "subfolder":"encoder",
    "encoder_max_length": 64,
    "insert_embs_token": "<insert_embs>",
    "insert_embs_token_id": -114,
    "insert_seq_token": "<insert_sep>",
    "st_config":{
      "architectures": [
        "BertModel"
      ],
      "attention_probs_dropout_prob": 0.1,
      "gradient_checkpointing": false,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 384,
      "initializer_range": 0.02,
      "intermediate_size": 1536,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 12,
      "num_hidden_layers": 6,
      "pad_token_id": 0,
      "position_embedding_type": "absolute",
      "transformers_version": "4.8.2",
      "type_vocab_size": 2,
      "use_cache": true,
      "vocab_size": 30522
    }
  },
  "projector_config": {
    "mlp_depth": 2,
    "encoder_hidden_size": 3584,
    "decoder_hidden_size": 3584,
    "num_heads": 1,
    "multihead": false
  }
}

"""
# longlin's model layer i adapt to the vllm
"""
# decoder prefix: 
decoder.model.xxxx

# encoder prefix
encoder.xxx

# projector prefix
projector.model.xxx
"""


# lly's Model (tablegpt_markup)
"""
├── added_tokens.json
├── config.json
├── configuration_codet5p.py
├── configuration_tablegpt.py
├── encoder # copied from  https://huggingface.co/Salesforce/codet5p-6b , and removed model file.
│   ├── README.md
│   ├── added_tokens.json
│   ├── config.json
│   ├── configuration_codet5p.py
│   ├── configuration_tablegpt.py
│   ├── merges.txt
│   ├── modeling_codet5p.py
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── generation_config.json
├── merges.txt
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── modeling_codet5p.py
├── modeling_tablegpt.py
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── trainer_state.json
├── utils.py
└── vocab.json
"""

# config.json
"""
{
  "architectures": [
    "TableGPTMarkupForCausalLM"
  ],
  "auto_map":{
    "AutoConfig": "configuration_tablegpt.TableGPTConfig",
    "AutoModel":"modeling_tablegpt.TableGPTChatModel"
  },
  "model_type": "tablegpt_markup",
  "torch_dtype": "bfloat16",
  "mlp_depth":1,
  "encoder_max_length": 2048,
  "placeholder_token":"<TABLE_CONTENT>",
  "placeholder_token_id": -114,
  "encoder_hidden_size": 1024,
  "decoder_hidden_size": 3584,
  "llm_config": {
    "_name_or_path": "Qwen/Qwen2-7B",
    "architectures": [
      "Qwen2ForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 131072,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 131072,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.41.2",
    "use_cache": false,
    "use_sliding_window": false,
    "vocab_size": 152064
  },
  "encoder_config": {
    "_name_or_path": "codet5p-350m-encoder",
    "activation_function": "gelu_new",
    "add_cross_attention": false,
    "architectures": [
      "CodeT5pModel"
    ],
    "subfolder":"encoder",
    "attn_pdrop": 0.0,
    "bad_words_ids": null,
    "bos_token_id": 1,
    "chunk_size_feed_forward": 0,
    "cross_attention_hidden_size": null,
    "decoder_start_token_id": 50256,
    "diversity_penalty": 0.0,
    "do_sample": false,
    "early_stopping": false,
    "embd_pdrop": 0.0,
    "encoder_no_repeat_ngram_size": 0,
    "eos_token_id": 50256,
    "exponential_decay_length_penalty": null,
    "finetuning_task": null,
    "forced_bos_token_id": null,
    "forced_eos_token_id": null,
    "gradient_checkpointing": false,
    "id2label": {
      "0": "LABEL_0",
      "1": "LABEL_1"
    },
    "initializer_range": 0.02,
    "is_decoder": false,
    "is_encoder_decoder": false,
    "label2id": {
      "LABEL_0": 0,
      "LABEL_1": 1
    },
    "layer_norm_epsilon": 1e-05,
    "length_penalty": 1.0,
    "max_length": 20,
    "min_length": 0,
    "model_type": "codet5p_module",
    "n_ctx": 2048,
    "n_embd": 1024,
    "n_head": 16,
    "n_inner": null,
    "n_layer": 20,
    "n_positions": 2048,
    "no_repeat_ngram_size": 0,
    "num_beam_groups": 1,
    "num_beams": 1,
    "num_return_sequences": 1,
    "output_attentions": false,
    "output_hidden_states": false,
    "output_scores": false,
    "pad_token_id": null,
    "prefix": null,
    "problem_type": null,
    "pruned_heads": {},
    "remove_invalid_values": false,
    "repetition_penalty": 1.0,
    "resid_pdrop": 0.0,
    "return_dict": true,
    "return_dict_in_generate": false,
    "rotary_dim": 32,
    "scale_attn_weights": true,
    "sep_token_id": null,
    "summary_activation": null,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": true,
    "summary_type": "cls_index",
    "summary_use_proj": true,
    "task_specific_params": null,
    "temperature": 1.0,
    "tf_legacy_loss": false,
    "tie_encoder_decoder": false,
    "tie_word_embeddings": false,
    "tokenizer_class": "GPT2Tokenizer",
    "top_k": 50,
    "top_p": 1.0,
    "torch_dtype": "float16",
    "torchscript": false,
    "transformers_version": "4.21.3",
    "typical_p": 1.0,
    "use_bfloat16": false,
    "use_cache": true,
    "vocab_size": 51200
  }
}

"""

# Please re-merge LLY's model and reset the layer parameter names as listed below.
# Also, add safe_serialization=True in your save_pretrained function.

"""
# decoder prefix: 
decoder.model.xxxx

# encoder prefix
encoder.xxx

# projector prefix
projector.xxx
"""


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
                        },
                    ],
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
    # (
    #     "/root/workspace/vllm/TV_show_data_versionskz.csv",
    #     "查看播放最好的电视剧类型",
    # ),
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
