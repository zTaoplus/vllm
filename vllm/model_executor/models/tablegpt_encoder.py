import gc
import typing as t

import numpy as np
import torch
from einops import rearrange
from torch import einsum, nn
from torch.nn import functional as F
from transformers import (AutoModel, BertConfig, PretrainedConfig,
                          PreTrainedTokenizer)

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import Table
from vllm.inputs import InputContext, LLMInputs
from vllm.multimodal.utils import cached_get_tokenizer

TABGPT_ENCODER = None


def load_encoder(config: PretrainedConfig):
    global TABGPT_ENCODER
    TABGPT_ENCODER = TableEncoder(config)
    return TABGPT_ENCODER


def get_embedded_table(table: Table, model_config: ModelConfig,
                       tokenizer: PreTrainedTokenizer):

    table_max_rows = model_config.hf_config.encoder_config.max_rows
    table_max_cols = model_config.hf_config.encoder_config.max_cols

    df_col_count = len(table["columns"])

    tb = np.array([tb_col["values"] for tb_col in table["columns"]])

    _, num_cols = tb.shape[0], tb.shape[1]

    if num_cols > table_max_rows:
        tb = tb[:, np.random.choice(num_cols, table_max_rows, replace=False)]
        num_cols = table_max_rows

    anchor_row_num = tb.shape[0]
    anchor_table = tb.reshape(-1)
    anchor_table = tokenizer(
        anchor_table.tolist(),
        padding='max_length',
        truncation=True,
        max_length=model_config.hf_config.encoder_max_length,
        return_tensors='pt')
    anchor_table = {
        k: v.reshape(anchor_row_num, num_cols, -1)
        for k, v in anchor_table.items()
    }

    num_cols = anchor_table['input_ids'].shape[1]

    anchor_table_row_num = anchor_table['input_ids'].shape[0]

    anchor_table_padded = {
        k: F.pad(v, (0, 0, 0, table_max_cols - v.shape[1], 0,
                     table_max_rows - v.shape[0]), "constant", 1)
        for k, v in anchor_table.items()
    }

    anchor_table_mask = np.zeros((table_max_rows, table_max_cols))

    anchor_table_mask[:anchor_table_row_num, :num_cols] = 1

    ret = (anchor_table_padded['input_ids'],
           anchor_table_padded['attention_mask'],
           anchor_table_padded['token_type_ids'],
           torch.tensor(anchor_table_mask), df_col_count)
    return ret


def get_encoder_output(tables: t.List[Table], model_config: ModelConfig,
                       tokenizer: PreTrainedTokenizer):

    table_count = [len(tables)]

    column_count = []
    table_embeds = []
    for table_list in [tables]:
        anchor_table_input_ids = []
        anchor_table_attention_mask = []
        anchor_table_token_type_ids = []
        anchor_table_mask = []
        cur_column_count = []
        for table in table_list:
            p, q, r, s, cnt = get_embedded_table(table, model_config,
                                                 tokenizer)
            cur_column_count.append(cnt)
            anchor_table_input_ids.append(p)
            anchor_table_attention_mask.append(q)
            anchor_table_token_type_ids.append(r)
            anchor_table_mask.append(s)

        column_count.append(cur_column_count)

        anchor_table_input_ids = torch.stack(
            anchor_table_input_ids, dim=0).to(device=TABGPT_ENCODER.st.device)
        anchor_table_attention_mask = torch.stack(
            anchor_table_attention_mask,
            dim=0).to(device=TABGPT_ENCODER.st.device)
        anchor_table_token_type_ids = torch.stack(
            anchor_table_token_type_ids,
            dim=0).to(device=TABGPT_ENCODER.st.device)
        anchor_table_mask = torch.stack(
            anchor_table_mask, dim=0).to(device=TABGPT_ENCODER.st.device)

        table_embeds.append(
            TABGPT_ENCODER(anchor_table_input_ids, anchor_table_attention_mask,
                           anchor_table_token_type_ids, anchor_table_mask))
        del (anchor_table_input_ids, anchor_table_attention_mask,
             anchor_table_token_type_ids, anchor_table_mask)
        torch.cuda.empty_cache()
        gc.collect()

    cat_table_embeds = [[] for _ in range(len(table_count))]
    for i in range(len(table_count)):
        for j in range(len(column_count[i])):
            cat_table_embeds[i].append(table_embeds[i][j, :column_count[i][j]])
        cat_table_embeds[i] = torch.cat(cat_table_embeds[i], dim=0)
    return cat_table_embeds


def input_processor_for_qwen2tb_encoder(ctx: InputContext,
                                        llm_inputs: LLMInputs):
    hf_config = ctx.model_config.hf_config
    if hf_config.encoder_config is None:
        raise ValueError(
            "Cannot found the table encoder config in the model hf config")

    mm_data = llm_inputs['multi_modal_data']

    tokenizer = cached_get_tokenizer(
        ctx.model_config.model,
        subfolder=ctx.model_config.hf_config.encoder_config.subfolder)

    table_embeddings = get_encoder_output(mm_data["table"],
                                          model_config=ctx.model_config,
                                          tokenizer=tokenizer)

    return LLMInputs(prompt_token_ids=llm_inputs["prompt_token_ids"],
                     multi_modal_data={"table": table_embeddings[0]})


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ff_encodings(x, B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# classes


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim * mult, dim))

    def forward(self, x, **kwargs):
        return self.net(x)


class RowColAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # s = batch size
        # b = number of rows
        # h = number of heads
        # n = number of columns
        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s b h n d', h=h),
                      (q, k, v))
        sim = einsum('s b h i d, s b h j d -> s b h i j', q, k) * self.scale

        # masking
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(
                1, sim.shape[1], sim.shape[2], 1, 1)
            sim = sim.masked_fill(mask == 0, float("-1e10"))

        attn = sim.softmax(dim=-1)
        out = einsum('s b h i j, s b h j d -> s b h i d', attn, v)
        out = rearrange(out, 's b h n d -> s b n (h d)', h=h)
        return self.to_out(out)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # s = batch size
        # b = number of rows
        # h = number of heads
        # n = number of columns
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
                      (q, k, v))

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # masking
        # torch.Size([12, 300, 300])
        if mask is not None:
            mask = (mask.unsqueeze(1).repeat(1, sim.shape[1], 1, 1))
            sim = sim.masked_fill(mask == 0, float("-1e10"))

        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class Qformer(nn.Module):

    def __init__(self, dim, dim_head, inner_dim, query_num):
        super().__init__()

        self.heads = inner_dim // dim_head
        self.query_num = query_num
        self.scale = dim_head**-0.5
        self.q = nn.Parameter(torch.randn(query_num, inner_dim))
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.ff = PreNorm(inner_dim, Residual(FeedForward(inner_dim)))

    def forward(self, x, mask=None):
        x = rearrange(x, 's b n d -> s n b d')

        h = self.heads
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q = self.q.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1,
                                                    1)
        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s b h n d', h=h),
                      (q, k, v))
        sim = einsum('s b h i d, s b h j d -> s b h i j', q, k) * self.scale

        # masking
        if mask is not None:
            mask = rearrange(mask, 's i j -> s j i')
            mask = mask[:, 0, :].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(
                1, sim.shape[1], sim.shape[2], sim.shape[3], 1)
            sim = sim.masked_fill(mask == 0, float("-1e10"))

        attn = sim.softmax(dim=-1)
        out = einsum('s b h i j, s b h j d -> s b h i d', attn, v)
        out = rearrange(out, 's b h n d -> s b n (h d)', h=h)

        out = self.ff(out)
        return out


class RowColTransformer(nn.Module):
    # dim = dim of each token
    # nfeats = number of features (columns)
    # depth = number of attention layers
    # heads = number of heads in multihead attention
    # dim_head = dim of each head
    def __init__(self,
                 dim,
                 nfeats,
                 depth,
                 heads,
                 dim_head,
                 attn_dropout,
                 ff_dropout,
                 style='col',
                 mask=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.style = style

        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(
                    nn.ModuleList([
                        PreNorm(
                            dim,
                            Residual(
                                RowColAttention(dim,
                                                heads=heads,
                                                dim_head=dim_head,
                                                dropout=attn_dropout))),
                        PreNorm(dim,
                                Residual(FeedForward(dim,
                                                     dropout=ff_dropout))),
                        PreNorm(
                            dim,
                            Residual(
                                RowColAttention(dim,
                                                heads=heads,
                                                dim_head=dim_head,
                                                dropout=attn_dropout))),
                        PreNorm(dim,
                                Residual(FeedForward(dim,
                                                     dropout=ff_dropout))),
                    ]))
            else:
                self.layers.append(
                    nn.ModuleList([
                        PreNorm(
                            dim * nfeats,
                            Residual(
                                Attention(dim * nfeats,
                                          heads=heads,
                                          dim_head=64,
                                          dropout=attn_dropout))),
                        PreNorm(
                            dim * nfeats,
                            Residual(
                                FeedForward(dim * nfeats,
                                            dropout=ff_dropout))),
                    ]))

    def forward(self, x, mask=None):
        _, _, n, _ = x.shape  # [bs, n_rows, n_cols, dim]
        row_mask = None
        col_mask = None
        if mask is not None:
            col_mask = einsum('b i j, b i k -> b j k', mask, mask)
            row_mask = einsum('b i j, b k j -> b i k', mask, mask)
        # print(col_mask.shape, row_mask.shape)
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x, mask=col_mask)
                x = ff1(x)
                x = rearrange(x, 's b n d -> s n b d')
                x = attn2(x, mask=row_mask)
                x = ff2(x)
                x = rearrange(x, 's n b d -> s b n d', n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 's b n d -> s 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 's 1 b (n d) -> s b n d', n=n)
        return x


# transformer
class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Residual(
                            Attention(dim,
                                      heads=heads,
                                      dim_head=dim_head,
                                      dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim,
                                                      dropout=ff_dropout))),
                ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


# mlp
class MLP(nn.Module):

    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):

    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(),
                                    nn.Linear(dims[1], dims[2]))

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def get_flatten_table_emb(table_emb, mask):
    flatten_table_emb = torch.zeros(table_emb.size(0), table_emb.size(2),
                                    table_emb.size(3)).to(table_emb.device)
    row_num = torch.sum(mask, dim=1).int()
    for i in range(len(table_emb)):
        flatten_table_emb[i] = torch.mean(table_emb[i, :row_num[i, 0], :, :],
                                          dim=0)
    return flatten_table_emb


# helpers
class sep_MLP(nn.Module):

    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


class TableEncoder(nn.Module):

    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__()
        self.config = config

        encoder_config = config.encoder_config

        self.num_cols = encoder_config.num_cols
        self.attentiontype = encoder_config.attentiontype
        # self.final_mlp_style = encoder_config.final_mlp_style
        self.pred_type = encoder_config.pred_type
        self.cont_dim = encoder_config.cont_dim
        self.pooling = encoder_config.pooling
        self.numeric_mlp = encoder_config.numeric_mlp
        self.ff_dropout = encoder_config.ff_dropout
        self.attn_dropout = encoder_config.attn_dropout
        self.dim_head = encoder_config.dim_head
        self.depth = encoder_config.depth
        self.heads = encoder_config.heads

        self.st = AutoModel.from_config(BertConfig(**encoder_config.st_config))
        self.dim = self.st.config.hidden_size

        self.st.pooler = None

        # transformer
        self.transformer = RowColTransformer(dim=self.dim,
                                             nfeats=self.num_cols,
                                             depth=self.depth,
                                             heads=self.heads,
                                             dim_head=self.dim_head,
                                             attn_dropout=self.attn_dropout,
                                             ff_dropout=self.ff_dropout,
                                             style=self.attentiontype)

        self.col_specific_projection_head = simple_MLP(
            [self.dim, self.dim, self.cont_dim])

        self.qformer = Qformer(dim=self.dim,
                               dim_head=128,
                               inner_dim=3584,
                               query_num=3)

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0]  #First element of model_output contains all token embeddings

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).to(dtype=token_embeddings.dtype)

        return torch.sum(token_embeddings * input_mask_expanded,
                         1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, input_ids, attention_mask, token_type_ids):
        bs, num_rows, num_cols, seq_len = input_ids.shape[0], input_ids.shape[
            1], input_ids.shape[2], input_ids.shape[3]
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(-1, seq_len)

        last_hidden_state = self.st(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        embeddings = embeddings.reshape(bs, num_rows, num_cols, -1)

        return embeddings

    @torch.inference_mode()
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        table_mask,
    ):

        tab_emb = self.get_embeddings(input_ids, attention_mask,
                                      token_type_ids)

        if self.pooling == 'cls':
            # roll the table on dim 1 (row dim)
            tab_emb = torch.roll(tab_emb, 1, 1)
            # insert [cls] token at the first row
            tab_emb[:, 0, :, :] = self.cls

        cell_emb = self.transformer(tab_emb, mask=table_mask)

        col_emb = self.attn_pooling(cell_emb, table_mask)

        return col_emb

    def attn_pooling(self, cell_emb, table_mask):
        output = self.qformer(cell_emb, mask=table_mask)
        return output
