# coding=utf-8
# Adapted from
from array import array
import random
import itertools
import string
from typing import Iterable, List, Optional, Tuple, Mapping, Dict

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, MultiModalConfig, ModelConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.quantization import QuantizationConfig
# from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import  Qwen2ForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.utils import cached_get_tokenizer

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import IntermediateTensors
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SequenceData
from vllm.transformers_utils.configs import (
    TableGPTMarkUpConfig,
    TableGPTContrastiveConfig,
)
from vllm.multimodal.base import MultiModalInputs

from .interfaces import SupportsMultiModal
from .tablegpt_encoder import (
    input_processor_for_tablegpt_encoder,
    load_encoder,
    dummy_data_for_contrastive_tablegpt,
    get_encoder_output,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    merge_multimodal_embeddings,
)

from vllm.model_executor.models.codet5_encoder import CodeT5pModel


_TABLE_TOKEN_COUNT_PER_COL = 3


def input_processor_for_contrastive_table(
    ctx: InputContext, llm_inputs: LLMInputs
):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "table" not in multi_modal_data:
        return llm_inputs

    return input_processor_for_tablegpt_encoder(ctx, llm_inputs)


# TODO: shold add the max table counts for the encoder?
def get_max_contrastive_table_tokens(ctx: InputContext):
    table_col_max_length = getattr(
        ctx.model_config.hf_config.encoder_config, "max_cols", 100
    )

    return table_col_max_length * _TABLE_TOKEN_COUNT_PER_COL


# encoder here?
def contrastive_table_input_mapper(ctx: InputContext, data: object):
    if isinstance(data, torch.Tensor):
        data = data.to(dtype=ctx.model_config.dtype)
        return  MultiModalInputs({"table": data})
    elif isinstance(data,(list,dict)):
            
        model_config = ctx.model_config
        tokenizer = cached_get_tokenizer(
            model_config.model,
            subfolder=model_config.hf_config.encoder_config.subfolder,
        )

        table_embeddings = get_encoder_output(
            data, model_config=model_config, tokenizer=tokenizer
        )

        
        return MultiModalInputs({"table": table_embeddings[0]})
    else:
        raise ValueError(f"Unexpected data type:{type(data)}")


@MULTIMODAL_REGISTRY.register_input_mapper(
    "table", contrastive_table_input_mapper
)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "table", get_max_contrastive_table_tokens
)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_contrastive_tablegpt)
@INPUT_REGISTRY.register_input_processor(input_processor_for_contrastive_table)
class TableGPTContrastiveForCausalLM(nn.Module, SupportsMultiModal):
    def __init__(
        self,
        config: TableGPTContrastiveConfig,
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

        # QWEN2 DIRECTLY
        # self.language_model = init_vllm_registered_model(
        #     config.text_config, cache_config, quant_config
        # )
        self.language_model = Qwen2ForCausalLM(config.text_config, 
                                               cache_config, 
                                               quant_config)

        if not self.config.encoder_config:
            raise ValueError("table encoder configs cannot found in hf config.")

        mlp_depth = self.config.projector_config.mlp_depth
        encoder_hidden_size = self.config.projector_config.encoder_hidden_size
        decoder_hidden_size = self.config.projector_config.decoder_hidden_size

        num_heads = self.config.projector_config.num_heads

        if not self.config.projector_config.multihead:
            num_heads = 1

        modules = [
            nn.Linear(encoder_hidden_size, decoder_hidden_size * num_heads)
        ]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(
                nn.Linear(
                    decoder_hidden_size * num_heads,
                    encoder_hidden_size * num_heads,
                )
            )

        self.projector = nn.Sequential(*modules)

        self.encoder = load_encoder(self.config)

    def _validate_get_table(self, **kwargs) -> torch.Tensor | None:
        table = kwargs.pop("table", None)
        if table is None or self.projector is None:
            return None

        return table

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        table_embeds = self._validate_get_table(**kwargs)

        if table_embeds is not None:
            cur_table_embeds = self.projector(table_embeds)

            cur_input_embeds = self.language_model.model.get_input_embeddings(
                input_ids.clamp(min=0)
            )

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                cur_input_embeds,
                cur_table_embeds,
                self.config.encoder_config.insert_embs_token_id,
            )

            input_ids = None

            del table_embeds, cur_table_embeds, cur_input_embeds

        else:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.language_model.logits_processor(
            self.language_model.lm_head, hidden_states, sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.language_model.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "projector.model.": "projector.",
                "decoder.": "language_model."
            })

        loader = AutoWeightsLoader(self)
        loader.load_weights(weights, mapper=hf_to_vllm_mapper)



def input_processor_for_markup_table(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "table" not in multi_modal_data:
        return llm_inputs

    placeholder_token_id = ctx.model_config.hf_config.placeholder_token_id
    max_length = ctx.model_config.hf_config.encoder_max_length

    tokenizer = cached_get_tokenizer(
        ctx.model_config.model,
        subfolder=ctx.model_config.hf_config.encoder_config.subfolder,
    )

    prompt_token_ids = llm_inputs["prompt_token_ids"]
    # # NOTE: only add to the head
    # prompt_token_ids = [placeholder_token_id] * max_length + prompt_token_ids

    # NOTE: add special id near the user query
    prompt_token_ids = torch.tensor(prompt_token_ids)

    placeholder_token_id = ctx.model_config.hf_config.placeholder_token_id
    max_length = ctx.model_config.hf_config.encoder_max_length

    indices = torch.where(prompt_token_ids == placeholder_token_id)[0]

    new_prompt_token_ids = None

    table_encoder_token_ids = tokenizer(
        multi_modal_data["table"],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids

    if len(indices) > 0:
        new_values = torch.tensor(
            [placeholder_token_id] * table_encoder_token_ids.shape[-1]
        )
        new_prompt_token_ids = torch.cat(
            (
                prompt_token_ids[: indices[0]],
                new_values,
                prompt_token_ids[indices[0] + 1 :],
            )
        )

    return LLMInputs(
        prompt_token_ids=new_prompt_token_ids.tolist()
        if new_prompt_token_ids is not None
        else prompt_token_ids.tolist(),
        multi_modal_data={"table": table_encoder_token_ids},
    )


def get_max_markup_table_tokens(ctx: InputContext) -> int:
    return ctx.model_config.hf_config.encoder_max_length


def dummy_tabledata_for_markup_tablegpt(model_config: ModelConfig) -> Dict:
    import pandas as pd

    def generate_random_text(length: int) -> str:
        """random string by length"""
        return "".join(
            random.choice(string.ascii_letters + string.digits)
            for _ in range(length)
        )

    encoder_max_lenth = model_config.hf_config.encoder_max_length

    # df with random cell string,length:20, cols count:100, row count: 50
    df = pd.DataFrame(
        {
            f"col_{i}": [generate_random_text(20) for _ in range(50)]
            for i in range(100)
        }
    )

    tokenizer = cached_get_tokenizer(
        model_config.model,
        subfolder=model_config.hf_config.encoder_config.subfolder,
    )

    table_encoder_token_ids: torch.Tensor = tokenizer(
        df.to_markdown(),
        return_tensors="pt",
        truncation=True,
        max_length=encoder_max_lenth,
    ).input_ids
    return {"table": table_encoder_token_ids}


def dummy_seq_data_for_markup_tablegpt(
    hf_config: TableGPTMarkUpConfig, seq_len: int
):
    placeholder_token_id = hf_config.placeholder_token_id
    encoder_max_length = hf_config.encoder_max_length

    # this is the table placeholder tokens for contrastive(lly) table encoder
    token_ids = array(
        VLLM_TOKEN_ID_ARRAY_TYPE, [placeholder_token_id] * encoder_max_length
    )

    # extend the token ids to max seq len
    token_ids += array(VLLM_TOKEN_ID_ARRAY_TYPE, [0]) * (
        seq_len - len(token_ids)
    )

    return SequenceData(token_ids)


def dummy_data_for_markup_tablegpt(
    ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]
):
    # num_tables = mm_counts["table"]
    hf_config = ctx.model_config.hf_config

    seq_data = dummy_seq_data_for_markup_tablegpt(
        hf_config, ctx.model_config.max_model_len
    )

    mm_data = dummy_tabledata_for_markup_tablegpt(ctx.model_config)

    return seq_data, mm_data


def markup_table_input_mapper(ctx: InputContext, data: object):
    if isinstance(data, torch.Tensor):
        return MultiModalInputs({"table": data.to(dtype=torch.int)})

    raise ValueError("mm data must be a torch tensor")


@MULTIMODAL_REGISTRY.register_input_mapper("table", markup_table_input_mapper)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "table", get_max_markup_table_tokens
)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_markup_tablegpt)
@INPUT_REGISTRY.register_input_processor(input_processor_for_markup_table)
class TableGPTMarkupForCausalLM(nn.Module, SupportsMultiModal):
    def __init__(
        self,
        config: TableGPTMarkUpConfig,
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
            config.text_config, cache_config, quant_config
        )

        if not self.config.encoder_config:
            raise ValueError("table encoder configs cannot found in hf config.")

        self.encoder = CodeT5pModel(self.config.encoder_config)

        encoder_hidden_size = self.config.encoder_hidden_size
        decoder_hidden_size = self.config.decoder_hidden_size

        modules = [nn.Linear(encoder_hidden_size, decoder_hidden_size)]

        for _ in range(1, self.config.mlp_depth):
            modules.append(nn.GELU())

            modules.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))

        self.projector = nn.Sequential(*modules)

    def _validate_get_table(self, **kwargs) -> torch.Tensor | None:
        table = kwargs.pop("table", None)
        if table is None or self.projector is None:
            return None

        return table

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        table_encoder_input_ids = self._validate_get_table(**kwargs)

        if table_encoder_input_ids is not None:
            table_embeds = self.encoder(
                input_ids=table_encoder_input_ids
            ).last_hidden_state
            cur_table_embeds = self.projector(table_embeds)

            cur_input_embeds = self.language_model.model.get_input_embeddings(
                input_ids.clamp(min=0)
            )

            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                cur_input_embeds,
                cur_table_embeds,
                self.config.placeholder_token_id,
            )

            del table_embeds

            input_ids = None

        else:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.language_model.compute_logits(
            hidden_states, sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.language_model.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "projector.": "projector.",
                "decoder.": "language_model.",
            })

        loader = AutoWeightsLoader(self)
        loader.load_weights(weights, mapper=hf_to_vllm_mapper)