import math
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
import torch.nn.functional as F

class AtomTokenEmbed(nn.Module):
    """原子种类的 embedding"""
    def __init__(self, num_types, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(num_types, emb_dim)

    def forward(self, zs):
        # zs: [B, L]
        return self.emb(zs)  # [B, L, emb_dim]


# -----------------------
# SAT-aware MultiHead Attention (causal)
# -----------------------
class SATCausalAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout=0.1, cond_dim=0, use_element_bias=False): 
        super().__init__() 
        self.n_head = n_head 
        self.d_head = d_head 
        self.d_model = d_model 
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_head)))
        self.q_proj = nn.Linear(d_model, n_head * d_head) 
        self.k_proj = nn.Linear(d_model, n_head * d_head) 
        self.v_proj = nn.Linear(d_model, n_head * d_head) 
        self.out_proj = nn.Linear(n_head * d_head, d_model) 
        self.out_proj._is_residual = True
        self.attn_drop = nn.Dropout(dropout) 
        self.proj_drop = nn.Dropout(dropout) 
        self.use_element_bias = use_element_bias 
        self.use_sat = False
        #结构 bias 可学习缩放 
        self.struct_scale = nn.Parameter(torch.zeros(1)) 
        self.cond_dim = cond_dim 
        if cond_dim != 0: self.lattice_proj = nn.Linear(cond_dim, self.n_head) 
        
        
    def forward( self, x, attn_mask=None ): 
        B, L, D = x.shape # QKV 
        q = self.q_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2) 
        k = self.k_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2) 
        v = self.v_proj(x).view(B, L, self.n_head, self.d_head).transpose(1, 2) 
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [B,H,L,L] 
        
        if attn_mask.dim()==2:
            mask = attn_mask[None,None,:,:]
        elif attn_mask.dim()==3:
            mask = attn_mask[:,None,:,:]
        scores = scores.masked_fill(mask, -1e9) #关键修改 3：softmax 数值稳定 
        
        attn = F.softmax(scores.float(), dim=-1).type_as(scores) 
        attn = self.attn_drop(attn) 
        out = torch.matmul(attn, v) 
        out = out.transpose(1, 2).contiguous().view(B, L, -1) 
        out = self.out_proj(out) 
        out = self.proj_drop(out) 
        #print(attn.mean(), attn.std())
        return out, attn

# Decoder block and Model
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_ff, cond_dim=0, dropout=0.1):#隐藏层维度、注意力头数、每个头的维度、前馈神经网络、前馈网络维度、dropout概率
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)#输入前的归一化
        self.attn = SATCausalAttention(d_model, n_head, d_head, dropout=dropout)#自注意力回归模块
        
        # ===== Cross-Attention（新增）=====
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        
        self.ln2 = nn.LayerNorm(d_model)#前馈网络前的归一化
        self.ff = nn.Sequential(#前馈网络
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )#两层线性+GELU激活
        self.ff[3]._is_residual = True
        self.ln_scale = nn.Parameter(torch.ones(1))
        

    def forward(self, x, condition, attn_mask=None, condition_mask=None,):
        h = self.ln1(x) * self.ln_scale
        
        a, attn_map = self.attn(h, attn_mask=attn_mask)
        x = x + a
        
        h = self.ln_cross(x)
        cross_out, cross_map = self.cross_attn(
            query=h,
            key=condition,
            value=condition,
            key_padding_mask=condition_mask
        )
        x = x + cross_out
        
        x = x + self.ff(self.ln2(x))
        return x, attn_map

def get_sinusoidal_embeddings(num_bins, d_model):
    embeddings = torch.zeros(num_bins, d_model)

    coord = torch.arange(0, num_bins, dtype=torch.float) / num_bins
    coord = coord.unsqueeze(1)  # [num_bins, 1]

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    embeddings[:, 0::2] = torch.sin(2 * math.pi * coord * div_term)
    embeddings[:, 1::2] = torch.cos(2 * math.pi * coord * div_term)

    return embeddings

class SATGPT(nn.Module):
    def __init__(self, num_atom_types, k_dim=0, d_model=256,
                 n_layer=6, n_head=8, d_head=32, d_ff=512,
                 num_bins=100, max_length=128,residual=False):

        super().__init__()

        self.d_model = d_model
        self.k_dim = k_dim
        self.num_bins = num_bins
        self.n_layer = n_layer

        self.atom_embed = nn.Embedding(num_atom_types+2, d_model)
        self.atom_embed_seq = nn.Embedding(num_atom_types+2, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.lattice_proj = nn.Linear(6, d_model)
        if k_dim > 0:self.cond_proj = nn.Linear(k_dim, d_model)
        else:self.cond_proj = None
        self.type_emb = nn.Embedding(3, d_model) # 0:lattice, 1:atom, 2:global_cond
        
        self.coord_embed = nn.Embedding(num_bins, d_model)
        self.bos_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.lm_head = nn.Linear(d_model, num_bins)
        
        self.pos_emb_atom = nn.Embedding(max_length, d_model)
        self.ln_pos = nn.LayerNorm(d_model)
        self.ln_atom = nn.LayerNorm(d_model)
        self.ln_lattice = nn.LayerNorm(d_model)
        self.ln_cond = nn.LayerNorm(d_model)
        self.ln_embed = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            GPTBlock(d_model, n_head, d_head, d_ff, k_dim)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # 🔥 残差层缩放
            if hasattr(module, "_is_residual"):
                std = 0.02 / math.sqrt(2 * self.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def quantize_coords(coords, num_bins=100):
        # coords: [B, L, 3] in [0,1]
        coords = coords.clamp(0, 0.9999)
        idx = (coords * num_bins).long()
        return idx  # [B, L, 3]
    
    def _condition_(self, Zs, lattice, cond=None):
        B, L = Zs.shape
        device = Zs.device

        atom_tokens = self.atom_embed(Zs)          # [B, L, D]
        atom_indices = torch.arange(L, device=device).unsqueeze(0) # [1, L]
        atom_tokens = atom_tokens + self.pos_emb_atom(atom_indices)
        atom_tokens = self.ln_atom(atom_tokens)
        atom_mask = (Zs == 0) 
        
        lattice_token = self.lattice_proj(lattice) 
        lattice_token = self.ln_lattice(lattice_token)
        lattice_token = lattice_token.unsqueeze(1)
        lattice_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        
        if cond is not None and self.cond_proj is not None:
            cond_token = self.cond_proj(cond) 
            cond_token = self.ln_cond(cond_token)
            cond_token = cond_token.unsqueeze(1) 
            cond_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            context = torch.cat([lattice_token, atom_tokens, cond_token], dim=1)
            context_mask = torch.cat([lattice_mask, atom_mask, cond_mask], dim=1)
        else:
            context = torch.cat([lattice_token, atom_tokens], dim=1)
            context_mask = torch.cat([lattice_mask, atom_mask], dim=1)
        
        context[:, 0:1, :] += self.type_emb(torch.tensor(0, device=device)) # Lattice
        context[:, 1:1+L, :] += self.type_emb(torch.tensor(1, device=device)) # Atoms
        if cond is not None:
            context[:, -1:, :] += self.type_emb(torch.tensor(2, device=device)) # Cond

        return context, context_mask

    def forward(self, Zs, coords, lengths, lattice,
                cond=None,
                mode="train",
                ):
        B, T  = Zs.shape
        device = Zs.device
        coord_idx = (coords.clamp(0, 0.9999) * self.num_bins).long()
        
        
        emb = self.coord_embed(coord_idx)
        emb[:, 0:1, :] = self.bos_embed
        
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = emb + self.pos_emb(pos)
        x = self.ln_embed(x)
        x = self.drop(x)
        
        condition, condition_mask = self._condition_(Zs, lattice, cond)
        
        def mask(lengths, B, L, device):
            pad_mask = pos >= lengths.unsqueeze(1)
            causal_mask = torch.triu(
                torch.ones(L, L, device=device, dtype=torch.bool),
                diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)
            key_mask = pad_mask[:, None, :].expand(-1, L, -1)
            query_mask = pad_mask[:, :, None].expand(-1, -1, L)
            attn_mask = causal_mask | key_mask | query_mask
            return attn_mask
        attn_mask = mask(lengths, B, T, device)
        
        x_struct = x
        for layer in self.layers:
            x_struct, _ = layer(
                x_struct,
                condition,
                attn_mask=attn_mask,
                condition_mask=condition_mask
            )

        x_struct = self.ln_f(x_struct)
        
        logits = self.lm_head(x_struct)
        
        if mode == "train":
            pred = logits[:, :-1]
            target = coord_idx[:, 1:]

            # 👉 mask
            target_mask = torch.arange(T-1, device=device).unsqueeze(0) < (lengths.unsqueeze(1) - 1)

            loss_raw = F.cross_entropy(
                pred.reshape(-1, self.num_bins),
                target.reshape(-1),
                reduction='none'
            ).reshape(B, T-1)
            loss = (loss_raw * target_mask).sum() / target_mask.sum().clamp(min=1)
            return logits, loss

        else:
            return logits, None
        
    def sample_logits(self, logits, temperature=1.0, top_k=None):
        """
        logits: [B, num_bins]
        """
        if temperature == 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            kth = v[:, -1].unsqueeze(-1)
            logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    @torch.no_grad()
    def generate(self, Zs, lengths, lattice, cond=None,
                temperature=0.8, top_k=5):
        B, T = Zs.shape  # T = 3L + 1
        device = Zs.device
        coord_seq = torch.zeros(B, 1, dtype=torch.float, device=device)
        for t in range(T - 1): 
            input_seq = torch.zeros(B, T, dtype=torch.float, device=device)
            input_seq[:, :coord_seq.shape[1]] = coord_seq
            logits, _ = self.forward(
                Zs=Zs,
                coords=input_seq,
                lengths=lengths,
                lattice=lattice,
                cond=cond,
                mode="infer"
            )
            logits_t = logits[:, coord_seq.shape[1]-1]
            next_token = self.sample_logits(
                logits_t, temperature, top_k
            ).unsqueeze(1)
            next_coord = next_token.float() / self.num_bins
            coord_seq = torch.cat([coord_seq, next_coord], dim=1)
        return coord_seq