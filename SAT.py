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
    No edges needed.
    """
    def __init__(self, embed_dim, num_layers=3, batch_norm=True):#节点表示的维度、消息传递层数（在子图中传播的步数）、是否在输出中使用归一化
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        # MP layers (coordinate-based)
        self.mps = nn.ModuleList([CoordMP(embed_dim) for _ in range(num_layers)])#创建数个消息传递层，把这些层注册到模块中，每个CoordMO输入输出维度都是embed_dim

        # output projection
        if batch_norm:
            self.bn = nn.BatchNorm1d(embed_dim * 2)#构建归一化层
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)#将拼接后的二维映射回去
        
    def forward(self, x, coords, 
                subgraph_node_index, 
                subgraph_indicator_index,
                sub_batch_index):
        
        # subgraph_node_index_now = 
        # subgraph_indicator_index_now = 
        # -------- 1. 获取子图节点信息 --------
        x_sub = x[subgraph_node_index]           # [S, D]
        coords_sub = coords[subgraph_node_index] # [S, 3]
        #print(x)

        # -------- 2. K 层 Message Passing --------
        h = x_sub
        coords_new = coords_sub
        for mp in self.mps:
            msg_h, coords_new = mp(h, coords_new)
            h = h + msg_h#在展开的子图节点作残差叠加

        # -------- 3. 聚合成每个中心节点的表示 --------
        # subgraph_indicator_index: 每个 subgraph node 属于哪个主节点
        #print(h,subgraph_indicator_index)
        h_pool = scatter_mean(h, subgraph_indicator_index, dim=0) #此处使用平均池化的方法
        # shape = [L, D]
        B, L_max = x.size(0), x.size(1)

        # 初始化 padded 结果
        h_pool_padded = torch.zeros(B, L_max, device=h_pool.device)

        # sub_batch_index: (L_real,)
        # subgraph_indicator_index: (L_real,)
        # h_pool: (L_real, D_h)

        h_pool_padded[:h_pool.size(0)] = h_pool

        #print(x, subgraph_node_index, subgraph_indicator_index)
        # -------- 4. 拼接 + 输出映射 --------
        #print(x, h.size(), subgraph_indicator_index.size(), h_pool_padded)
        out = torch.cat([x, h_pool_padded], dim=-1)

        if self.batch_norm:
            out = self.bn(out)

        out = self.out_proj(out)
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
        self.scale = 1.0 / math.sqrt(d_head)#缩放点积注意力
        self.q_proj = nn.Linear(d_model, n_head * d_head)
        self.k_proj = nn.Linear(d_model, n_head * d_head)
        self.v_proj = nn.Linear(d_model, n_head * d_head)#qkv投影
        self.out_proj = nn.Linear(n_head * d_head, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.use_element_bias = use_element_bias
        self.extractor = KHopCoordExtractor(embed_dim=self.d_model)
        # projections between model dim and extractor dim
        self.use_element_bias = use_element_bias

    def forward(self, x, coords, B_graph=None, subgraph_node_index=None, subgraph_indicator=None,sub_batch_index=None, attn_mask=None):
        # x: [B, L, d_model]
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_head, self.d_head).transpose(1,2)  # [B, H, L, d]
        k = self.k_proj(x).view(B, L, self.n_head, self.d_head).transpose(1,2)
        v = self.v_proj(x).view(B, L, self.n_head, self.d_head).transpose(1,2)
        # compute scores [B, H, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # scaled dot-prod
        
        x_flat = x.reshape(B*L, D)
        coords_flat = coords.reshape(B*L, 3)
        # 3) call extractor (expects flattened indices)
        # subgraph_node_index, subgraph_indicator are arrays indexing into flattened nodes
        if subgraph_node_index is not None:
            ex_out_flat = self.extractor(
                x_flat, coords_flat,
                subgraph_node_index=subgraph_node_index,
                subgraph_indicator_index=subgraph_indicator,
                sub_batch_index=sub_batch_index
            )
            x_struct = ex_out_flat.reshape(B, L, D)
            #coords_new = coords_new_flat.reshape(B, L, 3)

            # simple kernel example (dot product normalized)
            x_i = x_struct.unsqueeze(2)       # [B,L,1,d]
            x_j = x_struct.unsqueeze(1)       # [B,1,L,d]
            K = (x_i * x_j).sum(-1) / math.sqrt(x_struct.size(-1))  # [B,L,L]
            
            # === 结构感知特征（Coord + PE + MP） ===
            scores = scores +  K.unsqueeze(1)        # 扩展到 [B, H, L, L]

        # incorporate B_graph broadcast to heads
        if B_graph is not None:
            # B_graph: [B, L, L] -> expand to [B, H, L, L]
            scores = scores + B_graph.unsqueeze(1)

        # 自回归因果掩码 (attn_mask shape [L,L] or [B,L,L])
        if attn_mask is not None:
            # attn_mask expected shape [L, L] (broadcast) OR [B,L,L]
            if attn_mask.dim() == 2:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            else:
                scores = scores + attn_mask.unsqueeze(1)

        # attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [B, H, L, d_head] 加权得到输出
        out = out.transpose(1,2).contiguous().view(B, L, -1)  # [B, L, H*d]
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out, attn  # attn optional return for inspection

# -----------------------
# Decoder block and Model
# -----------------------
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_ff, dropout=0.1):#隐藏层维度、注意力头数、每个头的维度、前馈神经网络、前馈网络维度、dropout概率
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

    def forward(self, x, coords, B_graph=None, subgraph_node_index=None, subgraph_indicator=None , sub_batch_index=None, attn_mask=None):
        # x: [B,L,d_model]
        h = self.ln1(x)
        a, attn_map = self.attn(h, coords, B_graph=B_graph, subgraph_node_index=subgraph_node_index, subgraph_indicator=subgraph_indicator,sub_batch_index=sub_batch_index, attn_mask=attn_mask)
        x = x + a#残差连接
        x = x + self.ff(self.ln2(x))
        return x, attn_map

class SATGPT(nn.Module):
    def __init__(self, num_atom_types, y_dim, k_dim=0, d_model=256, n_layer=6, n_head=8, d_head=32, d_ff=512, coord_emb_dim=128, max_length=128):
        super().__init__()
        self.d_model = d_model
        self.k_dim = k_dim
        self.y_dim = y_dim
        
        #原子与坐标的嵌入
        self.atom_embed = AtomTokenEmbed(num_atom_types, d_model)#原子种类的嵌入
        self.coord_embed = CoordEmbed(in_dim=3, emb_dim=coord_emb_dim)#原子坐标的嵌入
        # project coord emb to model dim (we will sum atom emb and coord emb)
        self.coord_to_model = nn.Linear(coord_emb_dim, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)  # 位置编码：可准备把位置编码修改为SAT风格的
        self.drop = nn.Dropout(0.1)#随机丢弃
        
        #已知性质的CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if k_dim > 0:
            self.cond_proj = nn.Linear(k_dim, d_model)
            self.cond_proj_prop = nn.Linear(k_dim, d_model)
        else:
            self.cond_proj = None  # 允许“无条件”生成 / 预测
            self.cond_proj_prop = None
        
        # ===== Property Encoder (SAT-style, non-causal) =====
        self.prop_layers = nn.ModuleList(
            [GPTBlock(d_model, n_head, d_head, d_ff) for _ in range(n_layer)]
        )
        self.prop_ln = nn.LayerNorm(d_model)
        # 专用 CLS（不要和 decoder CLS 混）
        self.prop_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.prop_head = nn.Sequential( nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, y_dim), )
        

        self.layers = nn.ModuleList([GPTBlock(d_model, n_head, d_head, d_ff) for _ in range(n_layer)])#Transformer层
        self.ln_f = nn.LayerNorm(d_model)

        #输出头
        self.type_head = nn.Linear(d_model, num_atom_types)#预测类型
        self.coord_head = nn.Sequential(
            nn.Linear(d_model, d_ff//2),
            nn.ReLU(),
            nn.Linear(d_ff//2, 3)
        )#预测坐标

        # SAT偏置
        self.b_dist = RBFDistanceBias(num_centers=20, cutoff=8.0, emb_dim=64)#距离偏置
        self.b_elem = ElementPairBias(num_atom_types)#原子对偏置

        #SAT偏置的缩放因子
        self.beta = nn.Parameter(torch.tensor(1.0))
        
        self.coord_pe_mp = CoordPE_MP(
            emb_dim=d_model,     # 通常与 Transformer hidden size 一样
            k_steps=3,           # 消息传递步数（可调）
            pe_type='lap'        # lap / rw
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, Zs, coords, lengths, prop,cond=None, subgraph_node_index=None, subgraph_indicator=None, sub_batch_index=None, sat=False):
        """
        Zs: [B, L] (atom types)
        coords: [B, L, 3]
        lengths: [B] valid lengths
        Returns logits for each position (predict next token after each existing atom)
        We assume positions aligned: at index t (0-based), representation corresponds to atom t.
        """
        B, L = Zs.shape
        device = Zs.device
        
        pad_coords = torch.zeros(B, 1, 3, dtype=coords.dtype, device=device)  # [B,1,3]
        
        coords_in = torch.zeros_like(coords)
        coords_in[:, 1:] = coords[:, :-1]        # teacher forcing shift
        coords_pad = torch.cat([pad_coords, coords_in], dim=1)  # [B, L+1, 3]
        #token嵌入
        coord_e = self.coord_embed(coords_pad.detach())
        coord_e = self.coord_to_model(coord_e)
        
        # ===== Atom type embedding (NO SHIFT) =====
        atom_e = self.atom_embed(Zs)        # [B, L, d_model]
        atom_e = torch.cat(
            [torch.zeros(B, 1, self.d_model, device=device), atom_e],
            dim=1
        )                                   # [B, L+1, d_model]
        
        # ===== CLS token (conditioning only) =====
        cls = self.cls_token.expand(B, 1, self.d_model)   # [B,1,d_model]
        if cond is not None:
            cls = cls + self.cond_proj(cond).unsqueeze(1)  # inject known properties

        x = atom_e# + coord_e
        x[:, 0:1, :] = cls
        
        #重新
        _, Lp, _ = coords_pad.shape

        #位置嵌入
        pos = torch.arange(Lp, device=device).unsqueeze(0).expand(B, Lp)
        pos_e = self.pos_emb(pos)
        x = self.drop(x + pos_e)
        
        #有效位置的mask
        lengths_pad = lengths + 1
        mask = torch.arange(Lp, device=device).unsqueeze(0) < lengths_pad.unsqueeze(1)# [B,L] bool
        
        # 因果掩码
        causal = causal_mask(Lp, device=device)  # [Lp, Lp]
        # ---- FIX CLS semantics ----
        causal[:, 0] = 0
        causal[0, :] = 0

        #图全局偏置计算
        if sat:
            B_graph_struct = self.b_dist(coords_pad, mask=mask) \
                        + self.b_elem(Zs)
            B_graph_struct = B_graph_struct.masked_fill(
                ~mask.unsqueeze(1), 0
            ).masked_fill(
                ~mask.unsqueeze(2), 0
            )
            B_graph_struct = self.beta * B_graph_struct
            B_graph_masked = B_graph_struct + causal.unsqueeze(0)
        else:
            B_graph_struct = None
            B_graph_masked = causal.unsqueeze(0)
        
        # ===== Structure decoder (decoder-only) =====
        x_struct = x

        #attn_maps_struct = []
        for layer in self.layers:
            x_struct, attn_map = layer(
                x_struct,
                coords_pad,
                B_graph=B_graph_masked,
                subgraph_node_index=subgraph_node_index,
                subgraph_indicator=subgraph_indicator,
                sub_batch_index=sub_batch_index,
                attn_mask=causal        # causal only here
            )
            #attn_maps_struct.append(attn_map)

        x_struct = self.ln_f(x_struct)


        # ===== Structure prediction =====
        delta_preds = self.coord_head(x_struct)[:, 1:]   # [B,L,3]
        coords_base = coords_pad[:, 1:]                  # r_{t-1}
        coord_preds = coords_base + delta_preds          # 从预测坐标改为预测位移   

        # =========================
        # Property Encoder (SAT-style, non-causal)
        # =========================

        # ---- encoder input: CLS + atom tokens ----
        prop_cls = self.prop_cls.expand(B, 1, self.d_model)

        if cond is not None:
            prop_cls = prop_cls + self.cond_proj_prop(cond).unsqueeze(1)

        atom_tokens = x_struct[:, 1:, :]
        x_prop = torch.cat([prop_cls, atom_tokens], dim=1)

        # ---- mask ----
        lengths_prop = lengths + 1
        mask_prop = torch.arange(Lp, device=device).unsqueeze(0) < lengths_prop.unsqueeze(1)

        # ---- encoder forward (non-causal) ----
        for layer in self.prop_layers:
            x_prop, _ = layer(
                x_prop,
                coords_pad,
                B_graph=B_graph_struct,
                subgraph_node_index=subgraph_node_index,
                subgraph_indicator=subgraph_indicator,
                sub_batch_index=sub_batch_index,
                attn_mask=None          # ❗ 关键：非因果
            )

        x_prop = self.prop_ln(x_prop)

        # ---- CLS readout ----
        prop_preds = self.prop_head(x_prop[:, 0])   # [B, y_dim]

        # typically for autoreg: the target for position t is Z_{t+1}, r_{t+1}; so we may shift when computing losses
        return coord_preds, prop_preds#, {"struct": attn_maps_struct }
    
    @torch.no_grad()
    def sample(self, Zs, cond=None, max_len=None):
        B, L = Zs.shape
        device = Zs.device

        coords_gen = torch.zeros(B, L, 3, device=device)

        for t in range(L):
            coords_in = torch.zeros_like(coords_gen)
            if t > 0:
                coords_in[:, 1:t+1] = coords_gen[:, :t]

            coord_preds, _ = self.forward(
                Zs,
                coords_in,
                lengths=torch.full((B,), t+1, device=device),
                prop=None,
                cond=cond,
                subgraph_node_index=None, 
                subgraph_indicator=None, 
                sub_batch_index=None,
                sat=False
            )

            coords_gen[:, t] = coord_preds[:, t]

        return coords_gen
