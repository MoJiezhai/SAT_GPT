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

criterion = nn.L1Loss()

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

        pred_type = Z[:, :-1]
        target_type = Zs[:, 1:]

        type_mask = type_mask[:, 1:]

        correct_type = (
            pred_type == target_type
        ).float()

        accuracy_type = (
            correct_type * type_mask
        ).sum() / type_mask.sum().clamp(min=1)

        loss_type = (
            (1.0 - correct_type) * type_mask
        ).sum() / type_mask.sum().clamp(min=1)
    else: loss_type = torch.tensor(0.0, device=device)
    
    if coord_preds is not None:
        valid_mask = torch.zeros_like(coords[..., 0], dtype=torch.float) 
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

    if coord_preds is not None:
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
        

    else:
        rms_loss = torch.tensor(0.0, device=device)
        accuracy = 0

    return loss_type, loss_coord, loss_prop * prop_weight, rms_loss, accuracy
