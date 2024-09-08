import torch

from vllm.inputs.registry import InputContext
from vllm.logger import init_logger

from .base import MultiModalInputs, MultiModalPlugin

logger = init_logger(__name__)


class TablePlugin(MultiModalPlugin):
    """Plugin for table data."""

    def get_data_key(self) -> str:
        return "table"

    def _default_input_mapper(self, ctx: InputContext,
                              data: object) -> MultiModalInputs:
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"Table data must be a tensor, got {type(data)}")

        return MultiModalInputs({"table": data})

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        return 3000
