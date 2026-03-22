import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1, d_cond=None,activation="relu", layer_norm=True, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead ,batch_first=batch_first)
        self.layer_norm = layer_norm
        if layer_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        if not d_cond is None:
            self.d_cond = d_cond
            self.film1  = nn.Sequential(
    nn.Linear(d_cond, 512),
    nn.ReLU(),
    nn.Linear(512, 2*d_model)
)
            self.film2 = nn.Sequential(
    nn.Linear(d_cond, 512),
    nn.ReLU(),
    nn.Linear(512, 2*d_model)
)
            #print(d_cond)
    
    def forward(self, x, cond=None, attn_mask=None, key_padding_mask=None):
        if cond is not None:
            B, L = cond.shape
            d_cond = self.d_cond
            if L < d_cond:
                cond = F.pad(cond, (0, d_cond - L), value=0.0)
            elif L > d_cond:
                cond = cond[:, :d_cond]
            cond = cond.float()

            gamma_beta1 = self.film1(cond)
            gamma1, beta1 = gamma_beta1.chunk(2, dim=-1)

            gamma_beta2 = self.film2(cond)
            gamma2, beta2 = gamma_beta2.chunk(2, dim=-1)

        # --- Self-Attention ---
        x2 = self.norm1(x)
        if cond is not None:
            x2 = gamma1.unsqueeze(1) * x2 + beta1.unsqueeze(1)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(x2)

        # --- Feedforward ---
        x2 = self.norm2(x)
        if cond is not None:
            x2 = gamma2.unsqueeze(1) * x2 + beta2.unsqueeze(1)
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x2))))
        x = x + self.dropout(x2)

        return x

class TransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, cond=None, attn_mask=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, cond,attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x

class Transformer(nn.Module):
    def __init__(self,
            num_elements=118,
            max_atoms=40,
            num_cond=0,
            d_model=256,
            num_heads=8,
            dim_feedforward=1024,
            dropout=0.1,
            num_layers=8,
            predict_lattice=True,
            use_token_type=True):

        super().__init__()

        self.element_embed = nn.Embedding(num_elements, d_model, padding_idx=0)

        self.num_cond = num_cond
        if num_cond > 0:
            self.cond_embed = nn.Linear(1, d_model)

        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        #用于区分原子和性质
        self.use_token_type = use_token_type
        if use_token_type:
            # 0 = global
            # 1 = element
            # 2 = property
            self.type_embed = nn.Embedding(3, d_model)
        
        self.max_len = 1 + num_cond + max_atoms

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_len, d_model)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            d_cond=25
        )

        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        #Lattice Prediction Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 6)
        )
    
    def forward(self, Z, cond=None, lengths=None):
        device = Z.device
        B, Lmax = Z.shape

        output = self.element_embed(Z) 
        
        # Property Embedding
        if self.num_cond > 0 and cond is not None:
            cond_copy = cond.unsqueeze(-1)
            cond_tokens = self.cond_embed(cond_copy)
        else:
            cond_tokens = None
        
        # Global Token
        global_token = self.global_token.expand(B, -1, -1)

        if cond_tokens is not None:
            output = torch.cat((global_token, cond_tokens, output), dim=1)
        else:
            output = torch.cat((global_token, output), dim=1)

        total_len = output.size(1)
        
        output = output + self.pos_embed[:, :total_len]

        # 构造 padding mask
        # 原子 padding mask
        if lengths is not None:
            atom_mask = torch.arange(Lmax, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            atom_mask = (Z == 0)
        # global token 不 mask
        global_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        # cond tokens 不 mask
        if cond_tokens is not None:
            cond_mask = torch.zeros(B, cond_tokens.size(1), dtype=torch.bool, device=device)
            masks = [global_mask, cond_mask, atom_mask]  # 注意顺序对应拼接
        else:
            masks = [global_mask, atom_mask]
        key_padding_mask = torch.cat(masks, dim=1) 

        # Token Type Embedding
        if self.use_token_type:
            type_ids = torch.zeros(total_len, device=device)
            type_ids[0] = 0  # global token

            if cond_tokens is not None:
                type_ids[1:1+cond_tokens.size(1)] = 2  # property tokens
                type_ids[1+cond_tokens.size(1):] = 1   # element tokens
            else:
                type_ids[1:] = 1  # element tokens

            type_ids = type_ids.long().unsqueeze(0).expand(B, -1)
            output = output + self.type_embed(type_ids)
        
        #Transformer Encoder
        output = self.encoder(
            x=output,
            cond=Z.clone(),#None,
            key_padding_mask=key_padding_mask
        )
        

        # 取 global token
        z = output[:, 0]
        # 预测 lattice
        pred = self.head(z)

        return pred