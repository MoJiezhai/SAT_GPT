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
    n = len(cost_matrix)
    m = len(cost_matrix[0]) if n > 0 else 0
    size = max(n, m)
    big_num = 1e10  
    cost = [[big_num]*size for _ in range(size)]
    for i in range(n):
        for j in range(m):
            val = cost_matrix[i][j]
            if val is None: 
                val = big_num
            cost[i][j] = val

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

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

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
    coord_preds, prop_preds,
    Zs, coords, prop, lengths, 
    rms_threshold=0.5, prop_weight=1
):

    B, L, _ = coord_preds.shape
    device = coord_preds.device
    total_loss_coord = 0.0

    valid_mask = torch.zeros_like(coords[..., 0])  # [B, L]

    for i in range(B):
        valid_mask[i, :lengths[i]] = 1

    valid_mask = valid_mask.unsqueeze(-1)  # [B, L, 1]

    pred = coord_preds  
    target = coords 
    valid_mask = valid_mask
    loss = ((pred - target) ** 2) *  valid_mask
    denom = ( valid_mask).sum().clamp(min=1)
    loss_coord = loss.sum() / denom
    
    loss_prop = torch.tensor(0.0, device=device)
    if prop_preds is not None:

        for i in range(6):
            if i <=2:
                loss_prop += criterion(
                    prop_preds[:, i], 
                    prop[:, i] 
                )/10
            else:
                loss_prop += criterion(
                    prop_preds[:, i],
                    prop[:, i]
                )/180
       

        loss_prop = loss_prop 
        loss_prop /= 6

    rms_losses = []
    correct = 0
    total = 0
    for i in range(B):
        Li = lengths[i].item()
        coord_gt = coords[i, :Li] 
        coord_pr = coord_preds[i, :Li]
        diff = coord_pr - coord_gt
        diff = diff - torch.round(diff)
        shift = diff.mean(dim=0, keepdim=True)
        diff = diff - shift
        diff = diff - torch.round(diff)
        rms = torch.sqrt((diff ** 2).sum(dim=1).mean())
        rms_losses.append(rms)
        correct += (rms < rms_threshold).float().item()
        total += 1
    rms_loss = torch.stack(rms_losses).mean() if rms_losses else torch.tensor(0.0, device=device)
    accuracy = correct / max(total, 1)

    return loss_coord, loss_prop * prop_weight, rms_loss, accuracy
