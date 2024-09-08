# coding=utf-8
# Adapted from
import gc
import itertools
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers import AutoModel

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.multimodal.utils import cached_get_tokenizer


from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import (filter_weights, 
                                              init_vllm_registered_model,
                                              merge_multimodal_embeddings)
from vllm.model_executor.models.codet5_encoder import CodeT5pModel


def input_processor_for_table(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "table" not in multi_modal_data:
        return llm_inputs

    placeholder_token_id = ctx.model_config.hf_config.placeholder_token_id
    max_length = ctx.model_config.hf_config.encoder_max_length

    tokenizer = cached_get_tokenizer(ctx.model_config.model,
                                     subfolder=ctx.model_config.hf_config.encoder_config.subfolder)
    
    prompt_token_ids = llm_inputs["prompt_token_ids"]
    # # NOTE: only add to the head
    # prompt_token_ids = [placeholder_token_id] * max_length + prompt_token_ids
    
    # NOTE: add special id near the user query
    prompt_token_ids = torch.tensor(prompt_token_ids)

    placeholder_token_id = ctx.model_config.hf_config.placeholder_token_id
    max_length = ctx.model_config.hf_config.encoder_max_length

    indices = torch.where(prompt_token_ids == placeholder_token_id)[0]

    new_prompt_token_ids = None
    
    if len(indices) > 0:
        new_values = torch.tensor([placeholder_token_id] * max_length)
        new_prompt_token_ids = torch.cat((prompt_token_ids[:indices[0]], new_values, prompt_token_ids[indices[0] + 1:]))
        
    table_encoder_token_ids = tokenizer(multi_modal_data["table"], return_tensors="pt", truncation=True, max_length=max_length).input_ids
    
    return  LLMInputs(
        prompt_token_ids=new_prompt_token_ids.tolist() if new_prompt_token_ids is not None else prompt_token_ids.tolist(),
        multi_modal_data={"table":table_encoder_token_ids}
    )



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
        

        self.encoder = CodeT5pModel(self.config.encoder_config)

        encoder_hidden_size=self.config.encoder_hidden_size
        decoder_hidden_size=self.config.decoder_hidden_size
        
        modules = [nn.Linear(encoder_hidden_size, decoder_hidden_size)]
        
        for _ in range(1, self.config.mlp_depth):
            
            modules.append(nn.GELU())
            
            modules.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))
        
        self.projector = nn.Sequential(*modules)

    def _preprocess_table_embeddings(self, input_ids, table_embeds):

        cur_input_embeds = self.language_model.model.get_input_embeddings(input_ids.clamp(min=0))

        cur_table_embeds = self.projector(table_embeds) # forward through the projector

        
        mask = (input_ids == self.config.placeholder_token_id)
        indices = torch.nonzero(mask).squeeze()

        assert table_embeds.shape[0] == indices.shape[0], "Encoder embedding 行数应与 placeholder_token_id 的数量一致"

        
        # 依次替换 embedded_ids 中的 placeholder_token_id 部分
        for i, index in enumerate(indices):
            cur_input_embeds[index] = cur_table_embeds[i]


        return cur_input_embeds
    
    def _validate_get_table(self, **kwargs) -> torch.Tensor | None:

        table = kwargs.pop("table", None)
        if table is None or self.projector is None:
            return None

        # self.projector.to(device=table.device, dtype=table.dtype)

        return table

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs: object) -> torch.Tensor:
        table_encoder_input_ids = self._validate_get_table(**kwargs)

        if table_encoder_input_ids is not None:
            # torch.Size([1, 512, 1024]) should no add batch size
            table_embeds = self.encoder(input_ids=table_encoder_input_ids).last_hidden_state

            inputs_embeds = self._preprocess_table_embeddings(
                input_ids=input_ids,
                table_embeds=table_embeds.squeeze(0),
            )

            del table_embeds
            torch.cuda.empty_cache()
            gc.collect()

            input_ids = None

            # inputs_embeds = inputs_embeds.squeeze(0)

        else:
            inputs_embeds = None

        hidden_states = self.language_model.model(
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
        logits = self.language_model.compute_logits(hidden_states, sampling_metadata)
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
        
        # load llm model
        llm_weights = filter_weights(llm_weights, "decoder")
        self.language_model.load_weights(llm_weights)

        proj_weights = filter_weights(proj_weights, "projector")
        proj_params_dict = dict(self.projector.named_parameters())
        for name, loaded_weight in proj_weights:
            param = proj_params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        enc_weights = filter_weights(enc_weights, "encoder")
        enc_params_dict = dict(self.encoder.named_parameters())

        for name, loaded_weight in enc_weights:

            if name.endswith(".causal_mask"):
                continue

            param = enc_params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)