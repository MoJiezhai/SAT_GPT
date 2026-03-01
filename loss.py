import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def lattice6_to_matrix(lattice6):
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

criterion = nn.L1Loss()
criterion_coords = nn.L1Loss()


def autoreg_loss(
    coord_preds, prop_preds,
    Zs, coords, prop, lengths, 
    rms_threshold=0.5, prop_weight=1
):

    B, L, _ = coord_preds.shape
    device = coord_preds.device
    valid_mask = torch.zeros_like(coords[..., 0])  # [B, L]
    for i in range(B):
        valid_mask[i, :lengths[i]] = 1
    valid_mask = valid_mask.unsqueeze(-1)  # [B, L, 1]

    #coord
    loss = criterion_coords(coord_preds, coords)  # [B, L, 3]
    loss = loss * valid_mask
    loss_coord = loss.sum() / valid_mask.sum()

    #prop
    loss_prop = torch.tensor(0.0, device=device)
    if prop_preds is not None:
        for i in range(6):
            if i <=2:
                loss_prop += criterion(
                    prop_preds[:, i], 
                    prop[:, i] 
                )
            else:
                loss_prop += criterion(
                    prop_preds[:, i],  
                    prop[:, i]    
                )
        loss_prop = loss_prop 
        loss_prop /= 6

    # RMS
    rms_losses = []
    correct = 0
    total = 0
    for i in range(B):

        Li = lengths[i].item()
        if Li <= 1:
            continue

        coord_gt = coords[i, :Li] 
        coord_pr = coord_preds[i, :Li]

        #取消全局偏差
        diff = coord_pr - coord_gt
        diff = diff - torch.round(diff)
        shift = diff.mean(dim=0, keepdim=True)
        diff = diff - shift
        diff = diff - torch.round(diff)

        #rms计算
        rms = torch.sqrt((diff ** 2).sum(dim=1).mean())

        rms_losses.append(rms)
        correct += (rms < rms_threshold).float().item()
        total += 1

    rms_loss = torch.stack(rms_losses).mean() if rms_losses else torch.tensor(0.0, device=device)
    accuracy = correct / max(total, 1)

    return loss_coord, loss_prop * prop_weight, rms_loss, accuracy
