from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class TableGPTConfig(PretrainedConfig):
    model_type = 'tablegpt'
    is_composition = True

    def __init__(self,
                 encoder_config=None,
                 llm_config=None,
                 projector_config=None,
                 **kwargs):

        super().__init__(**kwargs)

        self.encoder_config = PretrainedConfig(**encoder_config)
        self.projector_config = PretrainedConfig(**projector_config)
        self.text_config = PretrainedConfig(**llm_config)