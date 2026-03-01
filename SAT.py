import math
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
import torch.nn.functional as F

# Model components
class CoordEmbed(nn.Module):
    """将三维坐标映射到 embedding 空间"""
    def __init__(self, in_dim=3, emb_dim=128, hidden=128, eps=1e-6):
        super().__init__()
        self.eps = eps#作为对数中的参数
        self.mlp = nn.Sequential(
            nn.Linear(in_dim+1, hidden),#将三维投影到隐藏维，输入维度加1接入对数感知
            nn.ReLU(),#非线性激活
            nn.Linear(hidden, emb_dim)#从隐藏维投影到嵌入维
        )

    def forward(self, coords):
        # coords: [B, L, 3]
        #return self.mlp(coords)  数据未增强时的状态
        # 半径 r
        r = torch.norm(coords, dim=-1, keepdim=True)      # [B, L, 1]
        # log 半径
        log_r = torch.log(r + self.eps)                   # [B, L, 1]
        # 单位方向向量
        direction = coords / (r + self.eps)               # [B, L, 3]
        # 拼接特征
        feat = torch.cat([log_r, direction], dim=-1)      # [B, L, 4]
        return self.mlp(feat)

class AtomTokenEmbed(nn.Module):
    """原子种类的 embedding"""
    def __init__(self, num_types, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(num_types, emb_dim)

    def forward(self, zs):
        # zs: [B, L]
        return self.emb(zs)  # [B, L, emb_dim]


# SAT-aware MultiHead Attention (causal)
class SATCausalAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout=0.1, use_element_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_head)

        self.q_proj = nn.Linear(d_model, n_head * d_head)
        self.k_proj = nn.Linear(d_model, n_head * d_head)
        self.v_proj = nn.Linear(d_model, n_head * d_head)
        self.out_proj = nn.Linear(n_head * d_head, d_model)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.use_element_bias = use_element_bias
        #self.extractor = KHopCoordExtractor(embed_dim=d_model)

        self.struct_scale = nn.Parameter(torch.zeros(1))
        
        lattice_dim = 6
        self.lattice_proj = nn.Linear(lattice_dim, self.n_head)

    def forward(
        self,
        x,
        lattice,
        B_graph=None,
        subgraph_node_index=None,
        subgraph_indicator=None,
        sub_batch_index=None,
        attn_mask=None
    ):
        B, L, D = x.shape

        # QKV
        q = self.q_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L,L]
        
        if lattice is not None:
            lattice_bias = self.lattice_proj(lattice)  # [B, H]
            lattice_bias = lattice_bias.unsqueeze(-1).unsqueeze(-1)
            scores = scores + lattice_bias

        # 结构特征注入
        if subgraph_node_index is not None:
            subgraph_node_index = None#我删去了这一部分，因为没有用到

        # 图结构 bias
        if B_graph is not None:
            scores = scores + B_graph.unsqueeze(1)

        # Mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                scores = scores + attn_mask.unsqueeze(1)
            else:
                raise ValueError("attn_mask must be [L,L] or [B,L,L]")

        attn = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out, attn 

# Decoder block and Model
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_ff, cond_dim=0, dropout=0.1):#隐藏层维度、注意力头数、每个头的维度、前馈神经网络、前馈网络维度、dropout概率
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)#输入前的归一化
        self.attn = SATCausalAttention(d_model, n_head, d_head, dropout=dropout)#自注意力回归模块
        self.ln2 = nn.LayerNorm(d_model)#前馈网络前的归一化
        self.ff = nn.Sequential(#前馈网络
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )#两层线性+GELU激活
        if cond_dim > 0 : self.film = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x, cond, lattice, B_graph=None, subgraph_node_index=None, subgraph_indicator=None , sub_batch_index=None, attn_mask=None):
        # x: [B,L,d_model]
        h = self.ln1(x)
        
        if not cond is None:#使用film调制（一般不用）
            #print(cond)
            gamma_beta = self.film(cond)  # [B, 2*d]
            gamma, beta = gamma_beta.chunk(2, dim=-1)

            gamma = gamma.unsqueeze(1)  # [B,1,d]
            beta  = beta.unsqueeze(1)

            h = gamma * h + beta
        
        a, attn_map = self.attn(h, lattice, B_graph=B_graph, subgraph_node_index=subgraph_node_index, subgraph_indicator=subgraph_indicator,sub_batch_index=sub_batch_index, attn_mask=attn_mask)
        x = x + a#残差连接
        x = x + self.ff(self.ln2(x))
        return x, attn_map

class SATGPT(nn.Module):
    def __init__(self, num_atom_types, k_dim=0, d_model=256,
                n_layer=6, n_head=8, d_head=32, d_ff=512,
                coord_emb_dim=128, max_length=128):
        super().__init__()
        self.d_model = d_model
        self.k_dim = k_dim

        # ===== Embeddings =====
        self.atom_embed = AtomTokenEmbed(num_atom_types, d_model)
        # self.coord_embed = CoordEmbed(in_dim=3, emb_dim=coord_emb_dim)
        # self.coord_to_model = nn.Linear(coord_emb_dim, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.drop = nn.Dropout(0.1)

        # ===== CLS =====
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if k_dim > 0:
            self.cond_proj = nn.Linear(k_dim, d_model)
        else:
            self.cond_proj = None

        # ===== Transformer =====
        self.layers = nn.ModuleList([
            GPTBlock(d_model, n_head, d_head, d_ff, k_dim)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # ===== Output heads =====
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.ReLU(),
            nn.Linear(d_ff // 2, 3)
        )


    def forward(self, Zs,  lengths, lattice,
                cond=None,
                subgraph_node_index=None, 
                subgraph_indicator=None, 
                sub_batch_index=None,
                teacher_forcing=False,
                noise_std=0.05):

        B, L = Zs.shape
        device = Zs.device
        
        atom_e = self.atom_embed(Zs)
        atom_e = torch.cat(
            [torch.zeros(B, 1, self.d_model, device=device), atom_e],
            dim=1
        )
        x = atom_e #+ coord_e

        #Positional encoding
        _, Lp, _ = atom_e.shape
        pos = torch.arange(Lp, device=device).unsqueeze(0).expand(B, Lp)
        x = self.drop(x + self.pos_emb(pos))
        
        # causal mask
        causal_mask = torch.triu(
            torch.ones(Lp, Lp, device=device),
            diagonal=1
        ).bool()

        # padding mask
        lengths_pad = lengths + 1
        pad_mask = (
            torch.arange(Lp, device=device).unsqueeze(0)
            >= lengths_pad.unsqueeze(1)
        )
        pad_mask_attn = pad_mask.unsqueeze(1).expand(-1, Lp, -1)
        # 合并 bool mask
        combined_bool = causal_mask.unsqueeze(0) | pad_mask_attn
        # 填充mask
        attn_mask = torch.zeros_like(combined_bool, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(combined_bool, -1e9)

        # Transformer decode
        x_struct = x
        for layer in self.layers:
            x_struct, _ = layer(
                x_struct,
                cond,
                lattice,
            )
        #x_struct = self.ln_f(x_struct) 这是我最近删除的调试
        coord_preds = self.coord_head(x_struct)[:, 1:]
        return coord_preds
    
    @torch.no_grad()
    def sample(self, Zs, lattice, cond=None):

        lengths = (Zs != 0).sum(dim=1)
        coord_preds = self.forward(
            Zs=Zs,
            lattice=lattice,
            cond=cond,
            lengths=lengths
        )

        return coord_preds