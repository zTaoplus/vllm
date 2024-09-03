# coding=utf-8
# Adapted from
import gc
import itertools
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors, SamplerOutput

from .interfaces import SupportsMultiModal
from .tablegpt_encoder import input_processor_for_qwen2tb_encoder, load_encoder
from .tablegpt_projector import MultiHeadProjector
from .utils import filter_weights, init_vllm_registered_model


def input_processor_for_table(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "table" not in multi_modal_data:
        return llm_inputs

    return input_processor_for_qwen2tb_encoder(ctx, llm_inputs)


@MULTIMODAL_REGISTRY.register_table_input_mapper()
@MULTIMODAL_REGISTRY.register_max_table_tokens()
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
