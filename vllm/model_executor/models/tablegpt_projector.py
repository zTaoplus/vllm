import re

from torch import nn
import torch
from transformers import PretrainedConfig


IGNORE_INDEX = -100

class MultiHeadProjector(nn.Module):

    # def __init__(self, projector_type, encoder_hidden_size, decoder_hidden_size, num_heads, torch_dtype=None, multihead = True, **kwargs):
    def __init__(self, config:PretrainedConfig):

        """
        Build a table projector based on the given configuration.

        Args:
            config (object): mm_projector_type: The type of projector to use. Defaults to 'linear'; hidden_size: ...
            **kwargs: Additional keyword arguments.

        Returns:
            object: The table projector.

        Raises:
            ValueError: If the projector type is unknown.
        """
            
        super().__init__()
        self.config = config
        projector_config = config.encoder_config.projector
        
        self.multihead = projector_config["multihead"]

        self.projector_type = projector_config["projector_type"]
        self.encoder_hidden_size = projector_config["encoder_hidden_size"]
        self.decoder_hidden_size = projector_config["decoder_hidden_size"]
        self.num_heads = projector_config["num_heads"]

        if not self.multihead:
            self.num_heads = 1
        

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', self.projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size * self.num_heads)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.decoder_hidden_size * self.num_heads, self.decoder_hidden_size * self.num_heads))
            self.model = nn.Sequential(*modules)
        else:
            raise ValueError(f'Unknown projector type: {self.projector_type}')
    
    def forward(self, x):
        ret = self.model(x)
        if self.multihead:
            ret = ret.view(*ret.shape[:-1], self.num_heads, -1)
        return ret
        
    def prepare_insert_embeds(
        self, *, decoder, input_ids, position_ids=None, attention_mask=None, past_key_values=None, labels=None, table_embeds, learnable_embeds = None
    ):
        assert learnable_embeds == None, "learnable embeddings is not yet supported"
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.

        new_input_embeds = []
        num_insert_embs = int((input_ids == self.config.encoder_config.insert_embs_token_id).sum())
        
        if num_insert_embs == 0:
            raise ValueError("No insert embs token found in the input_ids")
        
        cur_table_embeds = self(table_embeds)
        mask = input_ids == self.config.encoder_config.insert_embs_token_id
        consecutive = mask[:-2] & mask[1:-1] & mask[2:]
        indices = torch.nonzero(consecutive).flatten()
        insert_emb_token_indices = [-3] + indices.tolist() + [input_ids.shape[0]]
        
        
        cur_input_ids_noim = []
        
        for i in range(len(insert_emb_token_indices) - 1):
            cur_input_ids_noim.append(input_ids[insert_emb_token_indices[i]+3:insert_emb_token_indices[i+1]])
        
        split_sizes = [x.shape[0] for x in cur_input_ids_noim]
        cur_input_embeds = decoder.get_input_embeddings((torch.cat(cur_input_ids_noim)))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []


        n = num_insert_embs // 3
        for i in range(n + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            if i < n:
                cur_insert_emb_features = cur_table_embeds[i] # num_heads * decode_hidden
                if self.multihead:
                    assert cur_insert_emb_features.shape == (self.num_heads, self.decoder_hidden_size), f"not match: {cur_insert_emb_features.shape}, f{(self.num_heads), self.decoder_hidden_size}"
                cur_new_input_embeds.append(cur_insert_emb_features)
        
        device = next(decoder.parameters()).device
        cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        new_input_embeds.append(cur_new_input_embeds)

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)

        new_input_embeds_padded = []

        for i, cur_new_embed in enumerate(new_input_embeds):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config.text_config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        return  new_input_embeds