import math
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_mean
import torch.nn.functional as F

class AtomTokenEmbed(nn.Module):
    """原子种类的 embedding"""
    def __init__(self, num_types, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(num_types, emb_dim)

    def forward(self, zs):
        # zs: [B, L]
        return self.emb(zs)  # [B, L, emb_dim]

class KHopExtractor(nn.Module):

    def __init__(self, embed_dim, num_layers=1, batch_norm=True):
        super().__init__()

        self.num_layers = num_layers
        self.batch_norm = batch_norm

        # simple message passing layers
        self.mps = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)]
        )

        if batch_norm:
            self.bn = nn.BatchNorm1d(embed_dim * 2)

        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self,
        x,                         # [N_total, D]
        subgraph_node_index=None,       # [S]
        subgraph_indicator_index=None,  # [S]
        lengths=None
    ):
        N_total, D = x.shape
        device = x.device
        
        # -------------------------
        # 1️⃣ 取出子图节点
        # -------------------------
        x_sub = x[subgraph_node_index]   # [S, D]

        # -------------------------
        # 2️⃣ K-hop message passing
        # -------------------------
        h = x_sub

        for mp in self.mps:
            msg = mp(h)
            h = h + msg

        # -------------------------
        # 3️⃣ 聚合回中心节点
        # -------------------------
        h_pool = scatter_mean(
            h,
            subgraph_indicator_index,
            dim=0,
            dim_size=N_total
        )  # [N_total, D]

        # -------------------------
        # 4️⃣ 拼接原特征
        # -------------------------
        out = torch.cat([x, h_pool], dim=-1)  # [N_total, 2D]

        if self.batch_norm:
            out = self.bn(out)

        out = self.out_proj(out)  # [N_total, D]

        return out
# -----------------------
# SAT-aware MultiHead Attention (causal)
# -----------------------
class SATCausalAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout=0.1, cond_dim=0, use_element_bias=False): 
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
        self.extractor = KHopExtractor(embed_dim=d_model) 
        #结构 bias 可学习缩放 
        self.struct_scale = nn.Parameter(torch.zeros(1)) 
        self.cond_dim = cond_dim 
        if cond_dim != 0: self.lattice_proj = nn.Linear(cond_dim, self.n_head) 
    def forward( self, x, lattice, B_graph=None, subgraph_node_index=None, subgraph_indicator=None, sub_batch_index=None, attn_mask=None ): 
        B, L, D = x.shape # QKV 
        q = self.q_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2) 
        k = self.k_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2) 
        v = self.v_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2) 
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B,H,L,L] 
        # if lattice is not None: 
        # # lattice_feat = self.lattice_proj(lattice) # [B,H] # scores = scores + lattice_feat[:, :, None, None] 
        # # 结构特征注入 
        if subgraph_node_index is not None: 
            x_flat = x.reshape(B * L, D) 
            ex_out_flat = self.extractor( x_flat, subgraph_node_index=subgraph_node_index, subgraph_indicator_index=subgraph_indicator ) #print(1)
            x_struct = ex_out_flat.reshape(B, L, D) # efficient dot kernel 
            K = torch.matmul(x_struct, x_struct.transpose(1, 2)) / math.sqrt(D) 
            scores = scores + self.struct_scale * K.unsqueeze(1) #print(2) 
        # 图结构 bias 
        if B_graph is not None: 
            scores = scores + B_graph.unsqueeze(1) 
            
        if attn_mask is not None: 
            if attn_mask.dim() == 2: 
                mask = attn_mask.unsqueeze(0).unsqueeze(0) # [1,1,L,L] 
            elif attn_mask.dim() == 3: 
                mask = attn_mask.unsqueeze(1) # [B,1,L,L]
        else: 
            raise ValueError 
        scores = scores.masked_fill(mask, -1e4) #关键修改 3：softmax 数值稳定 
        
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

        # ===== Atom embedding =====
        self.atom_embed = AtomTokenEmbed(num_atom_types, d_model)

        # ===== Positional embedding =====
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.drop = nn.Dropout(0.1)

        # ===== Special tokens =====
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.lattice_proj = nn.Linear(6, d_model)

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

        # ===== Coordinate head =====
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.ReLU(),
            nn.Linear(d_ff // 2, 3)
        )

    def forward(self, Zs, lengths, lattice,
                cond=None,
                subgraph_node_index=None,
                subgraph_indicator=None,
                sub_batch_index=None,
                teacher_forcing=False,
                noise_std=0.05):

        B, L = Zs.shape
        device = Zs.device

        # Atom tokens
        atom_e = self.atom_embed(Zs)       # [B, L, d]

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)   # [B,1,d]

        # Lattice token
        lattice_token = self.lattice_proj(lattice).unsqueeze(1)  # [B,1,d]

        # Cond token
        if cond is not None and self.cond_proj is not None:
            cond_token = self.cond_proj(cond).unsqueeze(1)
        else:
            cond_token = torch.zeros(B, 1, self.d_model, device=device)

        # Token sequence
        x = torch.cat(
            [cls, lattice_token, cond_token, atom_e],
            dim=1
        )

        # Positional encoding
        Lp = x.shape[1]

        pos = torch.arange(Lp, device=device).unsqueeze(0).expand(B, Lp)

        x = self.drop(x + self.pos_emb(pos))

        # Padding mask
        prefix = 3
        lengths_pad = lengths + prefix
        pad_mask = (
            torch.arange(Lp, device=device).unsqueeze(0)
            >= lengths_pad.unsqueeze(1)
        )
        mask_k = pad_mask.unsqueeze(1).expand(-1, Lp, -1)
        mask_q = pad_mask.unsqueeze(2).expand(-1, -1, Lp)
        attn_mask = mask_q | mask_k

        # Transformer
        x_struct = x

        for layer in self.layers:

            x_struct, _ = layer(
                x_struct,
                cond,
                lattice, 
                subgraph_node_index=subgraph_node_index, subgraph_indicator=subgraph_indicator,sub_batch_index=sub_batch_index,
                attn_mask=attn_mask,
            )

        x_struct = self.ln_f(x_struct)

        # Coordinate prediction
        coord_preds = self.coord_head(x_struct)[:, 3:]

        return coord_preds