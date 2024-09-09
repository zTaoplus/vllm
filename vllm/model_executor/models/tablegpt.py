# coding=utf-8
# Adapted from
from array import array
import gc
import random
import string
import itertools
from typing import Iterable, List, Optional, Tuple, Mapping

import torch
from torch import nn
from transformers import PretrainedConfig
import pandas as pd

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SequenceData
from vllm.transformers_utils.configs import TableGPTConfig

from .interfaces import SupportsMultiModal
from .tablegpt_encoder import input_processor_for_qwen2tb_encoder, load_encoder
from .tablegpt_projector import MultiHeadProjector
from .utils import filter_weights, init_vllm_registered_model


def input_processor_for_table(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "table" not in multi_modal_data:
        return llm_inputs

    return input_processor_for_qwen2tb_encoder(ctx, llm_inputs)



def get_table_max_cols_rows(hf_config:TableGPTConfig) -> Tuple[int,int]:
    max_rows = hf_config.encoder_config.max_rows
    max_cols = hf_config.encoder_config.max_cols

    return max_cols,max_rows

def dummy_tabledata_for_tablegpt(
    hf_config: TableGPTConfig,
    num_tables: int,
    table_max_rows: Optional[int] = None,
    table_max_cols: Optional[int] = None,
):
    # make the table data mode
    # generated a table from max rows, max cols and encoder max length(cell length)

    def generate_random_text(length):
        """生成一个固定长度的随机字符串"""
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


    # 创建一个包含随机文本的DataFrame
    df = pd.DataFrame({
        f'col_{i}': [generate_random_text(hf_config.encoder_max_length) for _ in range(table_max_cols)]
        for i in range(table_max_rows)
    })

    # 转换成所需格式
    table = [
            {
                "columns": [
                    {
                        "name": df.columns[i],
                        "dtype": str(df.dtypes[i]),
                        "values": df[df.columns[i]].tolist()
                    }
                    for i in range(len(df.columns))
                ]
            }
        ]


    return {"table": table}


def dummy_seq_data_for_tablegpt(
    hf_config: TableGPTConfig,
    seq_len: int,
    num_tables: int,
):  
    
    encoder_config = hf_config.encoder_config

    table_token_insert_id = encoder_config.insert_embs_token_id
    encoder_table_max_col = encoder_config.max_cols

    # this is the table placeholder tokens for contrastive(longlin) table encoder
    token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                      [table_token_insert_id]) * 3 * encoder_table_max_col * num_tables
    
    # extend the token ids to max seq len
    token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE,
                       [0]) * (seq_len - len(token_ids))
    
    return SequenceData(token_ids)



def dummy_data_for_tablegpt(ctx: InputContext, seq_len: int,
                         mm_counts: Mapping[str, int]):
    
    # 1, make prompt str?
    # 2. use custom tokenize?
    # or make the input token ids directly
    num_tables = mm_counts["table"]
    hf_config = ctx.model_config.hf_config

    table_max_cols,table_max_rows = get_table_max_cols_rows()

    
    seq_data = dummy_seq_data_for_tablegpt(hf_config,ctx.model_config.max_model_len,num_tables)
    
    mm_data = ""

    return seq_data, mm_data


@MULTIMODAL_REGISTRY.register_table_input_mapper()
@MULTIMODAL_REGISTRY.register_max_table_tokens()
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_tablegpt)
@INPUT_REGISTRY.register_input_processor(input_processor_for_table)
class TableGPTForCausalLM(nn.Module, SupportsMultiModal):

    def __init__(
        self,
        config: PretrainedConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config
        self.lora_config = lora_config

        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            config.text_config, cache_config, quant_config)

        if not self.config.encoder_config:
            raise ValueError(
                "table encoder configs cannot found in hf config.")

        self.projector = MultiHeadProjector(self.config)

        self.encoder = load_encoder(self.config)

    def _validate_get_table(self, **kwargs) -> torch.Tensor | None:

        table = kwargs.pop("table", None)
        if table is None or self.projector is None:
            return None

        self.projector.model.to(device=table.device, dtype=table.dtype)

        return table

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs: object) -> torch.Tensor:
        table_embeds = self._validate_get_table(**kwargs)

        if table_embeds is not None:

            inputs_embeds = self.projector.prepare_insert_embeds(
                decoder=self.language_model.model,
                input_ids=input_ids,
                table_embeds=table_embeds,
            )
            del table_embeds
            torch.cuda.empty_cache()
            gc.collect()

            input_ids = None

            inputs_embeds = inputs_embeds.squeeze(0)

        else:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.language_model.logits_processor(
            self.language_model.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.language_model.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # prepare weight iterators for components
        enc_weights, proj_weights, llm_weights = itertools.tee(weights, 3)

        # load table encoder
        enc_weights = filter_weights(enc_weights, "encoder")
        enc_params_dict = dict(self.encoder.named_parameters())
        for name, loaded_weight in enc_weights:
            param = enc_params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        # load projector
        proj_weights = filter_weights(proj_weights, "projector")
        proj_params_dict = dict(self.projector.named_parameters())
        for name, loaded_weight in proj_weights:
            param = proj_params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        # load llm model
        llm_weights = filter_weights(llm_weights, "decoder")
        self.language_model.load_weights(llm_weights)
