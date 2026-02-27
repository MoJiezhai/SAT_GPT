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
from SATmain.models import GraphTransformer
from SATmain.data import GraphDataset
from SATmain.utils import count_parameters
from SATmain.position_encoding import POSENCODINGS
from SATmain.gnn_layers import GNN_TYPES
from timeit import default_timer as timer

def build_graph(coords):
    """
    coords: [N, 3]
    """
    N = coords.size(0)

    row, col = torch.meshgrid(
        torch.arange(N), torch.arange(N), indexing="ij"
    )
    edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)

    # 计算距离
    diff = coords[row.flatten()] - coords[col.flatten()]
    dist = torch.norm(diff, dim=1, keepdim=True)  # [E, 1]
    dist_int = (dist * 100).long()

    return edge_index, dist_int

# ---------------------------
# Dataset 封装（直接用内存）
# ---------------------------
class SATCrystalDataset(InMemoryDataset):
    def __init__(self, data_list, root, split='train'):
        self.split = split
        super().__init__(root)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices),
                   os.path.join(self.processed_dir, f'{split}.pt'))

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

# ---------------------------
# 核心转换函数
# ---------------------------
def convert_train_dataset(train_dataset):
    data_list = []

    for item in tqdm(train_dataset, desc='Converting dataset'):
        Z = item['Z'].long()
        coords = item['coords'].float()
        prop = item['prop'].float()

        edge_index, edge_attr = build_graph(coords)

        data = Data(
            x=Z.view(-1, 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=prop
        )
        data_list.append(data)
    
    dset = GraphDataset(data_list, degree=True, k_hop=2, se="gnn",
        use_subgraph_edge_attr=True)#下载数据集，从数据集中得到数据

    return dset

from torch_geometric.data import Data, Batch

def batch_to_pyg_data(batch, batch_size, device):
    data_list = []

    Z = batch['Z'].to(device)          # [B, N]
    coords = batch['coords'].to(device)  # [B, N, 3]
    lengths = batch['lengths'].to(device)
    prop = batch['prop'].to(device)

    B = Z.size(0)

    for i in range(B):
        n = lengths[i]

        Zi = Z[i, :n].long().detach()            # [n]
        coordsi = coords[i, :n].detach()         # [n, 3]
        propi = prop[i] # [P]

        edge_index, edge_attr = build_graph(coordsi)

        data = Data(
            x=Zi.view(-1, 1),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=propi.unsqueeze(0)
        )
        data = data.to(device)
        data_list.append(data)

    dset = GraphDataset(data_list, degree=True, k_hop=2, se="gnn",
        use_subgraph_edge_attr=True)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last=False)
    return loader



def prop_criterion(pred, target):

    pred = pred.view(-1, 3, 3)
    target = target.view(-1, 3, 3)

    # ---- scale = cubic root of volume ----
    volume = torch.abs(torch.det(target))      # (B,)
    scale = volume.pow(1/3).detach()           # 不参与梯度
    scale = scale.view(-1, 1)                  # (B,1)

    # ---- 向量误差 ----
    vec_loss = ((pred-target)**2).mean()


    # ---- 长度误差 ----
    pred_len = torch.norm(pred, dim=-1)
    true_len = torch.norm(target, dim=-1)
    len_err = torch.abs(pred_len - true_len)   # L1 更稳定
    len_loss = (len_err ).mean()
    
    loss = torch.log(vec_loss + 0.1 * len_loss)

    return loss



def proptrain(model, batch, batch_size , optimizer, lr_scheduler, iteration, device, use_cuda=False):
    model.train()

    loader = batch_to_pyg_data(batch, batch_size, device)
    running_loss = 0.0
    
    for i, data in enumerate(loader):
        
        size = len(data.y)
        # if lr_scheduler is not None:
        #     if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         pass  # 这里不动，等 epoch 结束再 step
        #     else:
        #         lr_scheduler.step()


        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = prop_criterion(output, data.y)
        #print(loss)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * size
        
        if iteration % 50 == 0:
            with torch.no_grad():
                print("pred mean:", output.mean().item())
                print("target mean:", data.y.mean().item())
                print("pred std:", output.std().item())
                print("target std:", data.y.std().item())


    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    return epoch_loss

def propval(model, batch, batch_size, device, use_cuda=False):
    model.eval()

    loader = batch_to_pyg_data(batch, batch_size, device)
    running_loss = 0.0
    out = []
    
    for i, data in enumerate(loader):
        
        size = len(data.y)
        # if lr_scheduler is not None:
        #     if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        #         pass  # 这里不动，等 epoch 结束再 step
        #     else:
        #         lr_scheduler.step()


        if use_cuda:
            data = data.cuda()

        output = model(data)
    return output

def build_corrected_lattice(prop_preds, coords, margin=0, delta_scale=1):
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

    coords_detached = coords.detach()
    coord_max = coords_detached.max(dim=1).values
    coord_min = coords_detached.min(dim=1).values


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
    I = torch.eye(3, device=coords.device).unsqueeze(0)
    transform = I + delta_L

    # =====================
    # 3️⃣ 最终晶胞
    # =====================

    lattice_final = L_orth @ transform

    return lattice_final.view(-1, 9)


