from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class CodeT5pModuleConfig(PretrainedConfig):
    model_type = "codet5p_module"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size=50400,
            n_positions=2048,
            n_ctx=2048,
            n_embd=4096,
            n_layer=28,
            n_head=16,
            rotary_dim=64,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            tie_word_embeddings=False,
            **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )




class TableGPTConfig(PretrainedConfig):
    model_type = 'tablegpt'
    is_composition = True

    def __init__(self,
                 encoder_config=None,
                 llm_config=None,
                 encoder_hidden_size=1024,
                 decoder_hidden_size=3584,
                 mlp_depth=1,
                 encoder_max_length=64,
                 placeholder_token="<TABLE_CONTNET>",
                 placeholder_token_id=-114,
                 **kwargs):

        super().__init__(**kwargs)

        self.encoder_config = CodeT5pModuleConfig(**encoder_config)
        self.text_config = PretrainedConfig(**llm_config)

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.mlp_depth = mlp_depth
        self.encoder_max_length = encoder_max_length
        self.placeholder_token= placeholder_token
        self.placeholder_token_id = placeholder_token_id
