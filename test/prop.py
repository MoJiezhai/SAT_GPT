import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                extract_zip)

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils
# from SATmain.models import GraphTransformer
# from SATmain.data import GraphDataset
# from SATmain.utils import count_parameters
# from SATmain.position_encoding import POSENCODINGS
# from SATmain.gnn_layers import GNN_TYPES
from timeit import default_timer as timer

def build_corrected_lattice(prop_preds, coords, margin=1e-5, delta_scale=1):
    """
    用预测坐标构造最小正交晶胞，并用 prop_preds 作为修正量。

    Parameters
    ----------
    prop_preds : Tensor [B, 9]
        模型输出的晶胞修正量

    coords : Tensor [B, L, 3]
        模型预测的原子坐标（无 padding）

    margin : float
        给正交晶胞增加的安全边界

    delta_scale : float
        修正量缩放因子（防止梯度爆炸）

    Returns
    -------
    lattice_final : Tensor [B, 9]
        最终用于 loss 的晶胞（展平成9维）
    """
    
    # =====================
    # 1️⃣ 最小正交晶胞
    # =====================

    coord_max = coords.max(dim=1).values   # [B,3]
    coord_min = coords.min(dim=1).values   # [B,3]

    orth_lengths = coord_max - coord_min + margin

    # 构造对角晶胞
    L_orth = torch.diag_embed(orth_lengths)

    # ❗关键：不允许梯度回传到坐标
    L_orth = L_orth.detach()

    # =====================
    # 2️⃣ 修正量（模型输出）
    # =====================

    delta_L = prop_preds.view(-1, 3, 3)

    # 稳定化（防止晶胞爆炸）
    delta_L = delta_scale * torch.tanh(delta_L)

    # =====================
    # 3️⃣ 最终晶胞
    # =====================

    lattice_final = L_orth + delta_L

    return lattice_final.view(-1, 9)
