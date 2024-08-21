import os

from torch import nn,einsum
import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizer
import numpy as np
from transformers import AutoModel,AutoTokenizer
from einops import rearrange
from vllm.envs import VLLM_TABLE_MAX_COLS,VLLM_TABLE_MAX_ROWS
import pandas as pd
import typing as t

IGNORE_INDEX = -100




from vllm.inputs import InputContext, LLMInputs

def get_embedded_table(df:pd.DataFrame, encoder_tokenizer:PreTrainedTokenizer):
    def process_table_df(table_df):
        numeric_columns = table_df.select_dtypes(include=["number"]).columns

        # fill missing values with mean
        table_df[numeric_columns] = table_df[numeric_columns].apply(
            lambda col: col.fillna(col.mean() if not col.isna().all() else 0)
        )
        if len(table_df) > VLLM_TABLE_MAX_ROWS:
            table_df = table_df.sample(n=VLLM_TABLE_MAX_ROWS)
            
        
        table_np = table_df.to_numpy().astype(str)
        
        return table_np
    
    def load_tokenized_table(anchor_table):
        anchor_table = process_table_df(anchor_table)
        _, num_cols = anchor_table.shape[0], anchor_table.shape[1]
        anchor_row_num = anchor_table.shape[0]
        anchor_table = anchor_table.reshape(-1)
        max_length = 64
        tokenized_anchor_table = encoder_tokenizer(anchor_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')                
        tokenized_anchor_table = {k: v.reshape(anchor_row_num, num_cols, -1) for k, v in tokenized_anchor_table.items()}
        return tokenized_anchor_table

    
    table_df = df

    df_col_count = table_df.shape[1]
    anchor_table = load_tokenized_table(table_df)
    num_cols = anchor_table['input_ids'].shape[1]
    anchor_table_row_num = anchor_table['input_ids'].shape[0]
    anchor_table_padded = {k: F.pad(v, (0, 0, 0, VLLM_TABLE_MAX_COLS - v.shape[1], 0, VLLM_TABLE_MAX_ROWS - v.shape[0]), "constant", 1) for k, v in anchor_table.items()}
    # print('..', anchor_table_padded['input_ids'].shape, anchor_table_padded['attention_mask'].shape, anchor_table_padded['token_type_ids'].shape)
    anchor_table_mask = np.zeros((VLLM_TABLE_MAX_ROWS, VLLM_TABLE_MAX_COLS))
    anchor_table_mask[:anchor_table_row_num, : num_cols] = 1
    ret = (
        anchor_table_padded['input_ids'],
        anchor_table_padded['attention_mask'],
        anchor_table_padded['token_type_ids'],
        torch.tensor(anchor_table_mask),
        df_col_count
    )
    return ret
        

def get_encoder_output(dfss:t.List[t.List[pd.DataFrame]],encoder:"TableEncoder",encoder_tokneizer:PreTrainedTokenizer):
    
    table_count = [len(c_list) for c_list in dfss]
    column_count = []
    table_embeds = []
    for c_list in dfss:
        anchor_table_input_ids = []
        anchor_table_attention_mask = []
        anchor_table_token_type_ids = []
        anchor_table_mask = []
        cur_column_count = []
        for c in c_list:
            p, q, r, s, cnt = get_embedded_table(c,encoder_tokneizer)
            cur_column_count.append(cnt)
            anchor_table_input_ids.append(p)
            anchor_table_attention_mask.append(q)
            anchor_table_token_type_ids.append(r)
            anchor_table_mask.append(s)
            
        column_count.append(cur_column_count)
        
        anchor_table_input_ids = torch.stack(anchor_table_input_ids, dim=0)
        anchor_table_attention_mask = torch.stack(anchor_table_attention_mask, dim=0)
        anchor_table_token_type_ids = torch.stack(anchor_table_token_type_ids, dim=0)
        anchor_table_mask = torch.stack(anchor_table_mask, dim=0)
        table_embeds.append(encoder(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask, inference=True))
        del anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask
    
    cat_table_embeds = [[] for _ in range(len(table_count))]
    for i in range(len(table_count)):
        for j in range(len(column_count[i])):
            cat_table_embeds[i].append(table_embeds[i][j, :column_count[i][j]])
        cat_table_embeds[i] = torch.cat(cat_table_embeds[i], dim = 0)
    return cat_table_embeds
    


def input_processor_for_qwen2tb_encoder(ctx:InputContext, llm_inputs:LLMInputs):
    hf_config = ctx.model_config.hf_config
    if hf_config.table_encoder is None:
        raise ValueError(
            "Cannot found the table encoder config in the model config"
        )

    mm_data = llm_inputs['multi_modal_data']

    dfs = [df for _, df in mm_data["table"]]
    
    tb_encoder = TableEncoder(**hf_config.table_encoder["encoder"])
    
    
    table_embeddings = get_encoder_output([dfs],tb_encoder,tb_encoder.tokenizer)

    return LLMInputs(
        prompt_token_ids=llm_inputs["prompt_token_ids"],
        multi_modal_data={"table":table_embeddings[0]}
    )

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ff_encodings(x,B):
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
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class RowColAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        # s = batch size
        # b = number of rows
        # h = number of heads
        # n = number of columns
        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s b h n d', h = h), (q, k, v))
        sim = einsum('s b h i d, s b h j d -> s b h i j', q, k) * self.scale

        # masking
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, sim.shape[1], sim.shape[2], 1, 1)
            sim = sim.masked_fill(mask == 0, float("-1e10"))

        attn = sim.softmax(dim = -1)
        out = einsum('s b h i j, s b h j d -> s b h i d', attn, v)
        out = rearrange(out, 's b h n d -> s b n (h d)', h = h)
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
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
        )
        
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # masking
        # torch.Size([12, 300, 300])
        if mask is not None:
            mask = (
                mask.unsqueeze(1)
                .repeat(1, sim.shape[1], 1, 1)
            )
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
        q = self.q.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1, 1)
        q, k, v = map(lambda t: rearrange(t, 's b n (h d) -> s b h n d', h = h), (q, k, v))
        sim = einsum('s b h i d, s b h j d -> s b h i j', q, k) * self.scale

        # masking
        if mask is not None:
            mask = rearrange(mask, 's i j -> s j i')
            mask = mask[:,0,:].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, sim.shape[1], sim.shape[2], sim.shape[3], 1)
            sim = sim.masked_fill(mask == 0, float("-1e10"))

        attn = sim.softmax(dim = -1)
        out = einsum('s b h i j, s b h j d -> s b h i d', attn, v)
        out = rearrange(out, 's b h n d -> s b n (h d)', h = h)
        
        out = self.ff(out)
        return out

class RowColTransformer(nn.Module):
    # dim = dim of each token
    # nfeats = number of features (columns)
    # depth = number of attention layers
    # heads = number of heads in multihead attention
    # dim_head = dim of each head
    def __init__(self, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col', mask=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.style = style

        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(RowColAttention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim, Residual(RowColAttention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))

    def forward(self, x, mask=None):
        _, _, n, _ = x.shape # [bs, n_rows, n_cols, dim]
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
                x = rearrange(x, 's n b d -> s b n d', n = n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 's b n d -> s 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 's 1 b (n d) -> s b n d', n = n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


# mlp
class MLP(nn.Module):
    def __init__(self, dims, act = None):
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
    def __init__(self,dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# main class

class TabAttention(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 1,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        lastmlp_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)

    def forward(self, x_categ, x_cont,x_categ_enc,x_cont_enc):
        device = x_categ.device
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim = -1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == 'MLP':
                x = self.transformer(x_categ_enc,x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else: 
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim = -1)                    
        flat_x = x.flatten(1)
        return self.mlp(flat_x)

class TableDecoder(nn.Module):
    def __init__(self, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col'):
        super().__init__()
        self.encoder = RowColTransformer(
            dim=dim,
            nfeats=nfeats,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            style=style
        )
        self.decoder = Transformer(
            dim=dim,
            # nfeats=nfeats,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        

    def forward(self, anchor_table_emb, shuffled_table, mask = None, anchor_mask = None):
        # 1. Encode shuffled table -> shuffled_table_emb: [bs, 2*n_cols, dim]
        # 2. Concat anchor_table_emb and shuffled_table_emb -> emb: [bs, 3*n_cols, dim]
        # 3. Segment emb?
        # 4. Decode emb with common transformer -> output: [bs, 3*n_cols, dim]
        shuffled_table_emb = self.encoder(shuffled_table, mask=mask) # [bs, n_rows, 2*n_cols, dim]
        
        shuffled_table_emb_ = get_flatten_table_emb(shuffled_table_emb, mask)
        
        joint_mask = torch.cat((anchor_mask[:,0,:], mask[:,0,:]), dim=1).unsqueeze(1)
        assert torch.sum(joint_mask) == torch.sum(anchor_mask[:,0,:]) + torch.sum(mask[:,0,:])
        joint_mask = einsum('b i j, b i k -> b j k', joint_mask, joint_mask)
        
        emb = torch.cat((anchor_table_emb, shuffled_table_emb_), dim=1)
        output = self.decoder(emb, joint_mask)
        return output

def get_flatten_table_emb(table_emb, mask):
    flatten_table_emb = torch.zeros(table_emb.size(0), table_emb.size(2), table_emb.size(3)).to(table_emb.device)
    row_num = torch.sum(mask, dim=1).int()
    for i in range(len(table_emb)):
        flatten_table_emb[i] = torch.mean(table_emb[i, :row_num[i,0], :, :], dim=0)
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
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TableEncoder, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        *,
        num_cols,
        depth,
        heads,
        dim_head,
        attn_dropout=0.0,
        ff_dropout=0.0,
        attentiontype="col",
        final_mlp_style="common",
        pred_type='generation',
        pooling='cls',
        col_name=False,
        model="all-MiniLM-L6-v2",
        numeric_mlp=False
    ):
        super().__init__()
        if not hasattr(self, 'initialized'):
            self.initialized = True
        
        self.num_cols = num_cols
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style
        self.pred_type = pred_type
        self.cont_dim = 256
        self.pooling = pooling
        self.numeric_mlp = numeric_mlp

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        # initialize sentence transformer
        # self.st_name = SENTENCE_TRANSFORMER
        self.model = model
        # model_dir = f"{MODELS_PATH}/{self.st_name}"
        self.st_name = model.rsplit("/",maxsplit=1)[-1]
        if self.st_name == 'all-MiniLM-L6-v2' or self.st_name == 'bge-small-en-v1.5':
            self.st = AutoModel.from_pretrained(self.model)
            self.dim = self.st.config.hidden_size
        elif self.st_name == 'puff-base-v1':
            vector_dim = 768
            self.dim = vector_dim
            self.st = AutoModel.from_pretrained(self.model)
            self.vector_linear = torch.nn.Linear(in_features=self.st.config.hidden_size, out_features=vector_dim)
            vector_linear_dict = {
                k.replace("linear.", ""): v for k, v in
                torch.load(os.path.join(self.model, f"2_Dense_{vector_dim}/pytorch_model.bin")).items()
            }
            self.vector_linear.load_state_dict(vector_linear_dict)
        else:
            raise ValueError("Invalid sentence transformer model")        
        
        for param in self.st.parameters():
            param.requires_grad = False
        self.st.pooler = None
        if self.numeric_mlp:
            self.num_mlp = simple_MLP([1, self.dim, self.dim])
        
        if self.pooling == 'cls':   
            self.cls = nn.Parameter(torch.randn(self.dim))

        # transformer
        if attentiontype == "col":
            self.transformer = Transformer(
                dim=self.dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
        elif attentiontype in ["row", "colrow"]:
            self.transformer = RowColTransformer(
                dim=self.dim,
                nfeats=num_cols,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype,
            )
            
        # projection head
        # needed for contrastive learning
        if self.pred_type == 'contrastive':
            self.col_specific_projection_head = simple_MLP([self.dim, self.dim, self.cont_dim])
            if col_name:
                self.col_name_projection_head = nn.Sequential(
                    Transformer(
                    dim=self.dim,
                    depth=1,
                    heads=heads,
                    dim_head=dim_head,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    ),
                    simple_MLP([self.dim, self.dim, self.cont_dim])
                )
        
        self.qformer = Qformer(dim=self.dim, dim_head=128, inner_dim=3584, query_num=3)
        
    
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(dtype = torch.bfloat16)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # input_size = [bs, num_rows, num_cols, seq_len]
    def get_embeddings(self, input_ids, attention_mask, token_type_ids):
        bs, num_rows, num_cols, seq_len = input_ids.shape[0], input_ids.shape[1], input_ids.shape[2], input_ids.shape[3]
        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape(-1, seq_len)
        
        if self.st_name == 'all-MiniLM-L6-v2':        
            last_hidden_state = self.st(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            embeddings = self.mean_pooling(last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        elif self.st_name == 'puff-base-v1':
            # puff version
            last_hidden_state = self.st(input_ids=input_ids, attention_mask=attention_mask)[0]
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            # mean pooling
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            embeddings = F.normalize(self.vector_linear(vectors), p=2, dim=-1)
        elif self.st_name == 'bge-small-en-v1.5':
            last_hidden_state = self.st(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # cls pooling
            embeddings = last_hidden_state[0][:,0]
            embeddings = F.normalize(embeddings, p=2, dim=-1)
                
        embeddings = embeddings.reshape(bs, num_rows, num_cols, -1)

        return embeddings
                
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        table_mask,
        inference=False
    ):
        if self.pred_type == 'contrastive':
            tab_emb = self.get_embeddings(input_ids, attention_mask, token_type_ids)
                        
            if self.pooling == 'cls':
                # roll the table on dim 1 (row dim)
                tab_emb = torch.roll(tab_emb, 1, 1)
                # insert [cls] token at the first row
                tab_emb[:,0,:,:] = self.cls
                        
            cell_emb = self.transformer(tab_emb, mask=table_mask)
            # [batch_size, num_cols, 384]
            
            if inference:
                for param in self.col_specific_projection_head.parameters():
                     param.requires_grad = True
                col_emb = self.attn_pooling(cell_emb, table_mask)
                return col_emb           
            elif self.pooling == 'cls':
                # the first row is cls -> cls pooling
                col_emb = cell_emb[:,0,:,:]
            else:
                # mean pooling
                col_emb = get_flatten_table_emb(cell_emb, table_mask)
            
            col_spe_cont_emb = F.normalize(self.col_specific_projection_head(col_emb), p=2, dim=-1)
            return col_spe_cont_emb
            
        else:
            x = self.transformer(input_ids, mask=table_mask)
            x = self.col_specific_projection_head(x) # [batch_szie, num_rows, num_cols, 384]
            output = get_flatten_table_emb(x, table_mask) # [batch_size, num_cols, 384]

            return output
        
    def unfreeze_st(self):
        for param in self.st.encoder.parameters():
            param.requires_grad = True
        if self.st_name == 'puff-base-v1':
            for param in self.vector_linear.parameters():
                param.requires_grad = True
                
    def attn_pooling(self, cell_emb, table_mask):
        output = self.qformer(cell_emb, mask=table_mask)
        return output
            