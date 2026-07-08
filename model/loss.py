import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import time


def lattice6_to_matrix(lattice6):
    """
    lattice6: (...,6) tensor [a,b,c, alpha, beta, gamma]
    angles in radians
    return: (...,3,3)
    """

    a, b, c, alpha, beta, gamma = lattice6.unbind(-1)
    
    alpha = alpha/180*math.pi
    beta = beta/180*math.pi
    gamma = gamma/180*math.pi

    cos_a = torch.cos(alpha)
    cos_b = torch.cos(beta)
    cos_g = torch.cos(gamma)
    sin_g = torch.sin(gamma)

    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)

    bx = b * cos_g
    by = b * sin_g
    bz = torch.zeros_like(a)

    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = torch.sqrt(torch.clamp(c*c - cx*cx - cy*cy, min=1e-12))

    L = torch.stack([
        torch.stack([ax, ay, az], dim=-1),
        torch.stack([bx, by, bz], dim=-1),
        torch.stack([cx, cy, cz], dim=-1)
    ], dim=-2)

    return L

def linear_sum_assignment(cost_matrix):
    """
    纯 Python 实现匈牙利算法（Kuhn-Munkres）
    输入: cost_matrix 为二维 list，形状 (n, m)
    输出: row_ind, col_ind 两个列表，对应原始矩阵的最优匹配
    """
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if n > 0 else 0
    size = max(n, m)
    big_num = 1e10  # 用于扩展虚拟行列

    # 扩展为方阵
    cost = [[big_num]*size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            val = cost_matrix[i][j]
            if val is None:  # 如果原矩阵有 None，可以用大数代替
                val = big_num
            cost[i][j] = val

    # 初始化
    u = [0.0] * size
    v = [0.0] * size
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        minv = [float('inf')] * (size + 1)
        used = [False] * (size + 1)
        j0 = 0

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0 - 1] - v[j - 1]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            # 更新标签
            for j in range(size + 1):
                if used[j]:
                    if p[j] > 0:
                        u[p[j] - 1] += delta
                    if j > 0:
                        v[j - 1] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        # 增广路径
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # 提取匹配结果（只保留原始矩阵的行列）
    row_ind = []
    col_ind = []
    for j in range(1, size + 1):
        if p[j] > 0:
            r = p[j] - 1
            c = j - 1
            if r < n and c < m:
                row_ind.append(r)
                col_ind.append(c)

    return row_ind, col_ind

criterion = nn.L1Loss()
criterion_coords = nn.L1Loss()
criterion_total = nn.L1Loss()

def autoreg_loss(
    Z, coord_preds, prop_preds,
    Zs, coords, prop, lengths, num_bins,
    rms_threshold=0.5, prop_weight=1
):

    B, L, _ = coords.shape
    device = coords.device

    if Z is not None:
        type_mask = torch.zeros_like(
            Zs,
            dtype=torch.float,
            device=Zs.device
        )

        for i in range(B):
            type_mask[i, :lengths[i]] = 1

        # =====================================================
        # shift
        # =====================================================
        pred_type = Z[:, :-1]
        target_type = Zs[:, 1:]

        type_mask = type_mask[:, 1:]

        # =====================================================
        # token match
        # =====================================================
        correct_type = (
            pred_type == target_type
        ).float()

        # =====================================================
        # masked accuracy
        # =====================================================
        accuracy_type = (
            correct_type * type_mask
        ).sum() / type_mask.sum().clamp(min=1)

        # =====================================================
        # token error loss
        # =====================================================
        loss_type = (
            (1.0 - correct_type) * type_mask
        ).sum() / type_mask.sum().clamp(min=1)
    else: loss_type = torch.tensor(0.0, device=device)
    
    if coord_preds is not None:
        valid_mask = torch.zeros_like(coords[..., 0], dtype=torch.float)  # [B, L]
        for i in range(B):
            valid_mask[i, :lengths[i]] = 1
        coord_preds_flat = coord_preds.reshape(coord_preds.size(0), -1)
        coords_flat = coords.reshape(coords.size(0), -1)
        coord_preds_idx = (
            coord_preds_flat.clamp(0, 0.9999) * num_bins
        ).long()
        coords_idx = (
            coords_flat.clamp(0, 0.9999) * num_bins
        ).long()
        coord_preds_idx = coord_preds_idx.clamp(0, num_bins - 1)
        coords_idx = coords_idx.clamp(0, num_bins - 1)
        valid_mask = (
            valid_mask.unsqueeze(-1)
            .expand(-1, -1, 3)
            .reshape(B, -1)
        )
        pred = coord_preds_idx[:, :-1]
        target = coords_idx[:, 1:]
        valid_mask = valid_mask[:, 1:]
        pred_logits = F.one_hot(
            pred,
            num_classes=num_bins
        ).float()
        loss = F.cross_entropy(
            pred_logits.reshape(-1, num_bins),
            target.reshape(-1),
            reduction='none'
        ).reshape(B, -1)
        loss = loss * valid_mask
        denom = valid_mask.sum().clamp(min=1)
        loss_coord = loss.sum() / denom
    else: loss_coord = torch.tensor(0.0, device=device)
    
    loss_prop = torch.tensor(0.0, device=device)
    if prop_preds is not None:
        for i in range(6):
            if i <=2:
                loss_prop += criterion(
                    prop_preds[:, i],   # [B,nbins]
                    prop[:, i]        # [B]
                )/10
            else:
                loss_prop += criterion(
                    prop_preds[:, i],   # [B,nbins]
                    prop[:, i]        # [B]
                )/180
       
        loss_prop = loss_prop 
        loss_prop /= 6

    if coord_preds is not None:
        rms_losses = []
        correct = 0
        total = 0
        for i in range(B):
            Li = lengths[i].item()
            coord_gt = coords[i, :Li]       # fractional GT
            coord_pr = coord_preds[i, :Li]  # fractional prediction
            diff = coord_pr - coord_gt
            diff = diff - torch.round(diff)     # PBC wrap first
            shift = diff.mean(dim=0, keepdim=True)
            diff = diff - shift
            diff = diff - torch.round(diff)     # wrap again after shift
            rms = torch.sqrt((diff ** 2).sum(dim=1).mean())
            rms_losses.append(rms)
            correct += (rms < rms_threshold).float().item()
            total += 1
        rms_loss = torch.stack(rms_losses).mean() if rms_losses else torch.tensor(0.0, device=device)
        accuracy = correct / max(total, 1)
        
        # L_pred = lattice6_to_matrix(prop_preds)
        # L_true = lattice6_to_matrix(prop)
        # C_pred = torch.matmul(coord_preds, L_pred)
        # C_true = torch.matmul(coords, L_true)
        # loss_total = criterion_total(C_pred, C_true)
        # loss_total = loss_total * valid_mask

        # loss_total = loss_total.sum() / valid_mask.sum()
    else:
        rms_loss = torch.tensor(0.0, device=device)
        accuracy = 0

    return loss_type, loss_coord, loss_prop * prop_weight, rms_loss, accuracy
