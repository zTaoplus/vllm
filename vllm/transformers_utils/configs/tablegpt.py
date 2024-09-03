from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class TableGPTConfig(PretrainedConfig):
    model_type = 'tablegpt'
    is_composition = True

    def __init__(
            self,
            encoder_config=None,
            llm_config=None,
            is_encoder_decoder=False,
            encoder_max_length=64,
            **kwargs):
        
        super().__init__(**kwargs)

        self.encoder_config = PretrainedConfig(**encoder_config)
        self.text_config = PretrainedConfig(**llm_config)
        
        self.is_encoder_decoder = is_encoder_decoder
        self.encoder_max_length = encoder_max_length