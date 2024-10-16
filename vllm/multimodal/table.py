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
        raise NotImplementedError("There is no default table input mapper")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        raise NotImplementedError(
            "There is no default maximum multimodal tokens")

