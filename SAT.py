import math
import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
import torch.nn.functional as F

class CoordEncoder(nn.Module):
    def __init__(self, emb_dim, num_rbf=16):
        super().__init__()
        # 原子类型 embedding
        self.atom_emb = nn.Embedding(118, emb_dim)  # 假设 Z < 118

        # RBF 基函数
        self.num_rbf = num_rbf
        centers = torch.linspace(0, 5.0, num_rbf)  # cutoff=5Å
        self.register_buffer("rbf_centers", centers)
        self.gamma = nn.Parameter(torch.tensor(10.0))

        # 将 RBF 编码投影到节点 embedding
        self.rbf_proj = nn.Linear(num_rbf, emb_dim)

        # 最终融合
        self.out_proj = nn.Linear(2 * emb_dim, emb_dim)

    def rbf(self, d):
        # d: [B,L,L]
        diff = d.unsqueeze(-1) - self.rbf_centers  # [B,L,L,K]
        return torch.exp(-self.gamma * diff**2)

    def forward(self, coords, Z):
        """
        coords: [B,L,3]
        Z:      [B,L]
        """
        B, L, _ = coords.shape

        # ① 原子 embedding
        h_atom = self.atom_emb(Z)  # [B,L,D]

        # ② 计算距离矩阵
        diff = coords[:, :, None, :] - coords[:, None, :, :]   # [B,L,L,3]
        dist = torch.norm(diff, dim=-1)                         # [B,L,L]

        # ③ RBF 编码
        rbf_feat = self.rbf(dist)                               # [B,L,L,K]
        rbf_local = rbf_feat.mean(dim=2)                        # [B,L,K]

        h_geo = self.rbf_proj(rbf_local)                        # [B,L,D]

        # ④ 融合
        h = torch.cat([h_atom, h_geo], dim=-1)                  # [B,L,2D]
        h = self.out_proj(h)                                    # [B,L,D]

        return h

class PositionEncoding(object):
    def compute_pe(self, graph):
        """单个图的PE, 留给子类实现"""
        raise NotImplementedError

    def batch_compute_pe(self, graphs):
        """
        graphs: list of graph objects, batch_size = B
        返回 [B, L_max, D] 张量，L_max是批次中节点数最大的图
        """
        pe_list = []
        max_nodes = max([g.num_nodes for g in graphs])

        for g in graphs:
            pe = self.compute_pe(g)  # [num_nodes, D]
            # pad到最大长度
            pad_len = max_nodes - g.num_nodes
            if pad_len > 0:
                pe = torch.cat([pe, torch.zeros(pad_len, pe.shape[1])], dim=0)
            pe_list.append(pe)

        batch_pe = torch.stack(pe_list, dim=0)  # [B, L_max, D]
        return batch_pe

class LapEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=True, normalization=None):
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        edge_index, edge_attr = utils.get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        L = utils.to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = np.real(EigVal[idx]), np.real(EigVec[:, idx])
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim + 1]).float()  # [num_nodes, D]

class RWEncoding(PositionEncoding):
    def __init__(self, dim, use_edge_attr=True, normalization=None):
        self.pos_enc_dim = dim
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        W0 = normalize_adj(graph.edge_index, num_nodes=graph.num_nodes).tocsc()
        W = W0
        vector = torch.zeros((graph.num_nodes, self.pos_enc_dim))
        vector[:, 0] = torch.from_numpy(W0.diagonal())
        for i in range(self.pos_enc_dim - 1):
            W = W.dot(W0)
            vector[:, i + 1] = torch.from_numpy(W.diagonal())
        return vector.float()

def normalize_adj(edge_index, edge_weight=None, num_nodes=None):
    edge_index, edge_weight = utils.remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1),
                                 device=edge_index.device)
    num_nodes = utils.num_nodes.maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = deg_inv[row] * edge_weight
    return utils.to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)

POSENCODINGS = {
    'lap': LapEncoding,
    'rw': RWEncoding,
}

class MP_Layer(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear_self = nn.Linear(emb_dim, emb_dim)
        self.linear_msg = nn.Linear(emb_dim, emb_dim)
        self.linear_update = nn.Linear(emb_dim * 2, emb_dim)
    
    def forward(self, h, coords, mask=None):
        """
        h: [B, L, D]   节点特征
        coords: [B, L, 3] 节点坐标
        mask: [B, L]   padding mask（可选）
        """

        B, L, D = h.shape

        # (1) 计算所有节点之间的 pairwise relative distance
        diff = coords[:, :, None, :] - coords[:, None, :, :]   # [B, L, L, 3]
        dist = torch.norm(diff, dim=-1)                        # [B, L, L]

        # (2) 用距离作为邻接权重（可改为更复杂核）
        # 越近越强，远的节点权重趋近0
        W = torch.exp(-dist)  # 高斯核，也可以用 1/dist 或 cutoff 等

        if mask is not None:
            # 把 padding 节点去掉
            W = W * mask[:, None, :] * mask[:, :, None]

        # (3) 聚合 message
        msg = torch.matmul(W, self.linear_msg(h))  # [B, L, D]

        # (4) update 节点特征
        h_new = torch.cat([self.linear_self(h), msg], dim=-1)
        h_new = self.linear_update(h_new)
        return h_new


#输出每个节点的结构感知embedding，用于构造SAT Attention bias的kernel
class CoordPE_MP(nn.Module):
    def __init__(self, emb_dim, k_steps, pe_type='lap'):#节点特征的维度、消息传递的步数、绝对位置编码的类型
        super().__init__()
        self.coord_encoder = CoordEncoder(emb_dim)#初始化位置编码，把节点的绝对坐标+原子类型编码成emd_dim维向量：用于观看坐标局部几何关系
        self.pe_encoder = POSENCODINGS[pe_type](dim=emb_dim)#选择一个位置编码模块（Lap或RWE），生成每个节点的绝对位置编码，捕捉节点在整个图的全局结构特征：用于图级别的位置结构
        self.mp_layers = nn.ModuleList([MP_Layer(emb_dim) for _ in range(k_steps)])#消息传递层，负责聚合邻居信息，更新节点特征

    def forward(self, coords, zs, graph=None):#节点坐标、节点类型、图对象（目前这个图还是需要边特征的）
        # 初始特征：坐标编码 + 原子类型 embedding
        h = self.coord_encoder(coords, zs)#将节点坐标、类型消息映射成emb_dim维度的特征向量，h是初始节点嵌入
        # PE embedding
        if graph != None : 
            pe = self.pe_encoder.compute_pe(graph).to(h.device)#绝对位置编码
            h = h + pe#将绝对位置编码加入初始节点特征
        # k-step MP
        for mp in self.mp_layers:#遍历所有的消息传递层
            h = mp(h, coords)#聚合邻居节点特征
        return h   # 返回节点的子图 embedding
#在我需要的Attention中加入
# subfeat = coord_pe_mp(coords, zs, graph)  # [B,L,D]
# K = kernel(subfeat)                        # [B,L,L]
# scores = qk + alpha * K[:,None,:,:]       # SAT bias

# -----------------------
# Utilities
# -----------------------
def causal_mask(seq_len, device):#因果掩码
    # upper triangular mask with -inf for future positions
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)
    return mask  # [seq_len, seq_len]

def rbf_expansion(distances, centers, gamma):#将距离转变成一组特征维度
    # distances: [..., N] or [..., N, N]
    # centers:   [C]
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers)**2)

# -----------------------
# Model components
# -----------------------
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

class RBFDistanceBias(nn.Module):
    """基于距离的 RBF -> MLP 偏置构造器，产生 B_graph 矩阵"""
    def __init__(self, num_centers=16, cutoff=8.0, emb_dim=64):
        super().__init__()
        self.num_centers = num_centers#中心个数
        self.cutoff = cutoff#距离截断
        centers = torch.linspace(0.0, cutoff, num_centers)#均匀采样中心，每个中心有高斯核函数
        self.register_buffer('centers', centers)
        self.gamma = 1.0 / ((centers[1] - centers[0])**2 + 1e-6)#表示RBF的尖锐度
        # small MLP to collapse RBF -> scalar bias
        self.mlp = nn.Sequential(#小MLP把RBF映射成Attention偏置
            nn.Linear(num_centers, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, coords, mask=None):
        # coords: [B, L, 3]
        # mask: [B, L] (bool) True where valid
        B, L, _ = coords.shape
        # compute pairwise distances
        # dist: [B, L, L]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # 计算位置差 [B, L, L, 3]
        dist = torch.norm(diff, dim=-1)  # 距离矩阵 [B, L, L]
        # optionally mask out large distances by setting to cutoff
        dist_clip = torch.clamp(dist, max=self.cutoff)#截断
        # RBF expansion
        rbf = torch.exp(-self.gamma * (dist_clip.unsqueeze(-1) - self.centers.view(1,1,1,-1))**2)  # [B,L,L,C]
        # collapse via mlp
        # reshape to [B*L*L, C]
        out = self.mlp(rbf.view(-1, self.num_centers))  # [B*L*L,1]
        out = out.view(B, L, L)  # [B, L, L] 得到标量偏置矩阵
        # mask invalid pairs (if provided)
        if mask is not None:
            # mask: True where valid positions; we want invalid pairs to have large -inf after scaling externally
            valid = mask.float()
            pair_valid = valid.unsqueeze(2) * valid.unsqueeze(1)  # [B,L,L]
            # Where pair_valid == 0, set a large negative value (we'll later add to scores)
            out = out * pair_valid + (1.0 - pair_valid) * (-1e9)
        return out  # [B, L, L] -- raw scalar bias per pair

class ElementPairBias(nn.Module):
    """元素对 embedding 偏置，可选"""
    def __init__(self, num_types, emb_dim=32):
        super().__init__()
        self.pair_emb = nn.Embedding(num_types * num_types, 1)#为每个类型对分配一个embedding索引
        self.num_types = num_types#保存类型数量以便计算

    def forward(self, zs):
        # zs: [B, L]
        B, L = zs.shape#解包出批次数和长度
        # compute pair index 为每个样本构建二维类型对矩阵
        zi = zs.unsqueeze(2).expand(B, L, L)  # [B,L,L]
        zj = zs.unsqueeze(1).expand(B, L, L)
        idx = zi * self.num_types + zj  # unique pair id 组合成唯一的pair id
        out = self.pair_emb(idx)  # [B,L,L,1] 使用embedding加入
        return out.squeeze(-1)

class CoordMP(nn.Module):
    """SE(3)-equivariant message passing (updates coords)."""
    def __init__(self, dim):
        super().__init__()
        self.phi_x = nn.Sequential(
            nn.Linear(dim*2 + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.phi_coord = nn.Sequential(
            nn.Linear(dim*2 + 1, 1),
            nn.SiLU()
        )

    def forward(self, x, coords):
        N, D = x.size()

        rel = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N,N,3]
        dist = torch.norm(rel, dim=-1, keepdim=True)     # [N,N,1]

        x_i = x.unsqueeze(1).expand(-1, N, -1)
        x_j = x.unsqueeze(0).expand(N, -1, -1)

        input_ij = torch.cat([x_i, x_j, dist], dim=-1)

        # --- 1. feature update (invariant)
        msg_x = self.phi_x(input_ij).mean(dim=1)

        # --- 2. coordinate update (equivariant)
        w = self.phi_coord(input_ij)                     # [N,N,1]
        delta_coords = (w * rel).mean(dim=1)             # [N,3]
        coords_new = coords + delta_coords

        return msg_x, coords_new


class KHopCoordExtractor(nn.Module):
    """
    K-hop subgraph extractor using ONLY coords + Zs + subgraph indexing.
    Works on flattened nodes: x shape [N_total, D]
    """
    def __init__(self, embed_dim, num_layers=3, batch_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        self.mps = nn.ModuleList(
            [CoordMP(embed_dim) for _ in range(num_layers)]
        )

        if batch_norm:
            self.bn = nn.BatchNorm1d(embed_dim * 2)

        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(
        self,
        x,                         # [N_total, D]
        coords,                    # [N_total, 3]
        subgraph_node_index,       # [S]
        subgraph_indicator_index,  # [S]  -> each sub node belongs to which center
        sub_batch_index=None
    ):
        N_total, D = x.shape

        # -------------------------
        # 1️⃣ 取出子图节点
        # -------------------------
        x_sub = x[subgraph_node_index]           # [S, D]
        coords_sub = coords[subgraph_node_index] # [S, 3]

        # -------------------------
        # 2️⃣ K-hop message passing
        # -------------------------
        h = x_sub
        coords_new = coords_sub

        for mp in self.mps:
            msg_h, coords_new = mp(h, coords_new)
            h = h + msg_h

        # -------------------------
        # 3️⃣ 聚合回中心节点
        # -------------------------
        # ⭐ 必须指定 dim_size
        h_pool = scatter_mean(
            h,
            subgraph_indicator_index,
            dim=0,
            dim_size=N_total   # 保证输出长度固定
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
        self.extractor = KHopCoordExtractor(embed_dim=d_model)

        #结构 bias 可学习缩放
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
            # lattice: [B, lattice_dim]
            lattice_bias = self.lattice_proj(lattice)  # [B, H]
            lattice_bias = lattice_bias.unsqueeze(-1).unsqueeze(-1)
            # [B, H, 1, 1]
            scores = scores + lattice_bias

        # 结构特征注入
        if subgraph_node_index is not None:
            x_flat = x.reshape(B * L, D)
            ex_out_flat = self.extractor(
                x_flat,
                None,
                subgraph_node_index=subgraph_node_index,
                subgraph_indicator_index=subgraph_indicator,
                sub_batch_index=sub_batch_index
            )
            x_struct = ex_out_flat.reshape(B, L, D)
            # dot kernel
            x_i = x_struct.unsqueeze(2)  # [B,L,1,D]
            x_j = x_struct.unsqueeze(1)  # [B,1,L,D]
            K = (x_i * x_j).sum(-1) / math.sqrt(D)  # [B,L,L]
            #关键修改 2：加入可学习缩放
            scores = scores + self.struct_scale * K.unsqueeze(1)

        # 图结构 bias
        if B_graph is not None:
            scores = scores + B_graph.unsqueeze(1)

        # Mask 处理（更安全）
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                scores = scores + attn_mask.unsqueeze(1)
            else:
                raise ValueError("attn_mask must be [L,L] or [B,L,L]")

        #关键修改 3：softmax 数值稳定
        attn = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out, attn # attn optional return for inspection

# -----------------------
# Decoder block and Model
# -----------------------
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
        
        if not cond is None:
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

        #Teacher forcing for coords
        # coords_in = coords.clone()
        # if teacher_forcing:
        #     coords_noisy = coords + noise_std * torch.randn_like(coords)
        #     coords_in[:, 1:] = coords_noisy[:, :-1]
        # else:
        #     coords_in[:, 1:] = coords[:, :-1]

        # #Embedding
        # pad_coords = torch.zeros(B, 1, 3, device=device)
        # coords_pad = torch.cat([pad_coords, coords_in], dim=1)
        # coord_e = self.coord_embed(coords_pad)
        # coord_e = self.coord_to_model(coord_e)
        
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

        # ⭐ 转成加法 mask
        attn_mask = torch.zeros_like(combined_bool, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(combined_bool, -1e9)

        # STEP 7 — Transformer decode
        x_struct = x
        for layer in self.layers:
            x_struct, _ = layer(
                x_struct,
                #coords_pad,
                cond,
                lattice,
                # subgraph_node_index=subgraph_node_index, 
                # subgraph_indicator=subgraph_indicator, 
                # sub_batch_index=sub_batch_index,
                # attn_mask=attn_mask,
            )

        x_struct = self.ln_f(x_struct)

        coord_preds = self.coord_head(x_struct)[:, 1:]
        #print(x_struct.std(dim=1))
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