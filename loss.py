import torch
import torch.nn.functional as F
import torch.nn as nn
import math


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

class LatticeDiscretizer:
    def __init__(self, n_bins=256):
        self.n_bins = n_bins

        # length range (unchanged)
        self.len_min = 1.0
        self.len_max = 20.0

        # angle range → 改成弧度
        self.ang_min = math.radians(60.0)
        self.ang_max = math.radians(120.0)

    def encode(self, lattice6):
        """
        lattice6: [B,6]
        angles MUST be in radians
        return: [B,6] int labels
        """

        lengths = lattice6[:, :3]
        angles = lattice6[:, 3:]
        
        if angles.max() > 2 * math.pi:
            angles = torch.deg2rad(angles)

        # normalize to 0-1
        l = (lengths - self.len_min) / (self.len_max - self.len_min)
        a = (angles - self.ang_min) / (self.ang_max - self.ang_min)

        l = torch.clamp(l, 0, 1)
        a = torch.clamp(a, 0, 1)

        bins_l = (l * (self.n_bins - 1)).long()
        bins_a = (a * (self.n_bins - 1)).long()

        return torch.cat([bins_l, bins_a], dim=-1)

    def decode(self, bins):
        bins = bins.float()

        bins_l = bins[:, :3]
        bins_a = bins[:, 3:]

        # length decode
        l = (bins_l + 0.5) / (self.n_bins - 1)
        lengths = self.len_min + l * (self.len_max - self.len_min)

        # angle decode (returns radians)
        a = (bins_a + 0.5) / (self.n_bins - 1)
        angles = self.ang_min + a * (self.ang_max - self.ang_min)

        return torch.cat([lengths, angles], dim=-1)

def soft_decode(prop_logits, discretizer):
    """
    连续可导的晶格解码（替代 argmax）

    prop_logits: [B, 6, n_bins]
    discretizer: 你的离散器对象

    return:
        lat6_pred: [B, 6]  (连续值)
    """
    B, P, n_bins = prop_logits.shape
    device = prop_logits.device

    # --- softmax 得到概率 ---
    probs = F.softmax(prop_logits, dim=-1)   # [B,6,n_bins]

    # --- bin index 向量 ---
    bin_ids = torch.arange(n_bins, device=device).float()  # [n_bins]

    # --- 计算期望 bin ---
    bins_continuous = (probs * bin_ids).sum(dim=-1)  # [B,6]

    # --- decode 成真实晶格值 ---
    lat6_pred = discretizer.decode(bins_continuous)

    return lat6_pred

discretizer = LatticeDiscretizer(n_bins=256)
criterion = nn.L1Loss()
criterion_coords = nn.L1Loss()


def autoreg_loss(
    coord_preds, prop_preds,
    Zs, coords, prop, lengths, 
    rms_threshold=0.5, prop_weight=1
):

    B, L, _ = coord_preds.shape
    device = coord_preds.device

    # ==============================
    # coordinate loss (不变)
    # ==============================

    valid_mask = torch.zeros_like(coords[..., 0])  # [B, L]

    for i in range(B):
        valid_mask[i, :lengths[i]] = 1

    valid_mask = valid_mask.unsqueeze(-1)  # [B, L, 1]

    loss = criterion_coords(coord_preds, coords)  # [B, L, 3]
    loss = loss * valid_mask

    loss_coord = loss.sum() / valid_mask.sum()

    # ==============================
    # NEW: lattice classification loss
    # ==============================
    loss_prop = torch.tensor(0.0, device=device)

    if prop_preds is not None:

        # prop_preds: [B,6,nbins]
        # prop:       [B,6] continuous

        #labels = discretizer.encode(prop).to(device)   # [B,6]
        
        #print(prop,labels)

        for i in range(6):
            if i <=2:
                loss_prop += criterion(
                    prop_preds[:, i],   # [B,nbins]
                    prop[:, i]        # [B]
                )
            else:
                loss_prop += criterion(
                    prop_preds[:, i],   # [B,nbins]
                    prop[:, i]        # [B]
                )
            #print(i, loss_prop.item())
        def lattice_log_loss(pred, target):
            eps = 1e-6

            pred_len = pred[:, :3]
            true_len = target[:, :3]

            loss_log = (torch.log(pred_len + eps) - torch.log(true_len + eps))**2
            
            lengths = pred[:, :3]

            mean_len = lengths.mean(dim=1, keepdim=True)
            loss_mean = ((lengths - mean_len) / mean_len)**2
            
            # pred_vol = lattice_volume(lattice_pred)
            # true_vol = lattice_volume(lattice_true)
            
            return loss_mean.mean()
        def length_relative_range_loss(pred,
                               target,
                               lower_ratio=0.2,
                               upper_ratio=1.3):

            pred_len = pred[:, :3]
            true_len = target[:, :3]

            min_len = true_len * lower_ratio
            max_len = true_len * upper_ratio

            lower_violation = F.relu(min_len - pred_len)
            upper_violation = F.relu(pred_len - max_len)
            
            mean_len = pred_len.mean(dim=1, keepdim=True)  # [B,1]

            mean_violation = F.relu(
                (mean_len * 0.5) - pred_len
            )

            loss = lower_violation**2 + upper_violation**2 + 5*mean_violation**2
            return loss.mean()
        def angle_range_loss(
            lattice_pred,
            min_angle=60.0,
            max_angle=120.0,
            margin=8.0,
            edge_push_weight=0.05,
            hard_weight=1.0,
            center=90.0,
            center_tol=1.0,
            center_weight=0.2,
        ):

            angles = lattice_pred[:, 3:]

            # -----------------------
            # 1️⃣ 出界强惩罚
            # -----------------------
            lower_violation = F.relu(min_angle - angles)
            upper_violation = F.relu(angles - max_angle)
            hard_loss = lower_violation**2 + upper_violation**2

            # -----------------------
            # 2️⃣ 贴边排斥
            # -----------------------
            left_edge = F.relu((min_angle + margin) - angles)
            right_edge = F.relu(angles - (max_angle - margin))

            left_push = (left_edge / margin) ** 2
            right_push = (right_edge / margin) ** 2
            edge_loss = left_push + right_push

            # -----------------------
            # 3️⃣ 弱 90° 偏置（仅在远离90时生效）
            # -----------------------
            center_diff = torch.abs(angles - center)

            # 只有当偏离超过 tolerance 才惩罚
            center_violation = F.relu(center_diff - center_tol)

            # 归一化避免梯度过大
            center_loss = (center_violation / center_tol) ** 2

            # -----------------------
            # 总 loss
            # -----------------------
            loss = (
                hard_weight * hard_loss
                + edge_push_weight * edge_loss
                + center_weight * center_loss
            )

            return loss.mean()
        def lattice_volume(lattice):
            a, b, c = lattice[:, 0], lattice[:, 1], lattice[:, 2]
            alpha, beta, gamma = lattice[:, 3], lattice[:, 4], lattice[:, 5]

                # 转弧度
            alpha = torch.deg2rad(alpha)
            beta  = torch.deg2rad(beta)
            gamma = torch.deg2rad(gamma)

            cos_a = torch.cos(alpha)
            cos_b = torch.cos(beta)
            cos_g = torch.cos(gamma)

            volume_term = (
                1
                + 2*cos_a*cos_b*cos_g
                - cos_a**2
                - cos_b**2
                - cos_g**2
            )

            volume = a * b * c * torch.sqrt(torch.clamp(volume_term, min=1e-8))

            return volume
        def volume_loss(prop_preds,
                prop,
                log_tol=0.4,
                min_volume=5.0,
                hard_weight=1.0,
                band_weight=1.0):

            pred_vol = lattice_volume(prop_preds)
            true_vol = lattice_volume(prop)

                # ---------------------------
                # 1️⃣ log 相对误差（scale invariant）
                # ---------------------------
            log_diff = torch.log(pred_vol + 1e-8) - torch.log(true_vol + 1e-8)

                # 允许一定浮动范围
            band_violation = F.relu(torch.abs(log_diff) - log_tol)
            band_loss = band_violation**2

                # ---------------------------
                # 2️⃣ 体积下界（防 collapse）
                # ---------------------------
            hard_violation = F.relu(min_volume - pred_vol)
            hard_loss = hard_violation**2

            loss = band_weight * band_loss + hard_weight * hard_loss

            return loss.mean()
        
        loss_prop = loss_prop #+volume_loss(prop_preds, prop) +angle_range_loss(prop_preds) +length_relative_range_loss(prop_preds, prop)*10 + lattice_log_loss(prop_preds, prop)*100 
        loss_prop /= 6
    # def lattice_log_loss(pred, target):
    #     eps = 1e-8
    #     pred_len = pred[:, :3]
    #     true_len = target[:, :3]

    #     return ((torch.log(pred_len + eps) - 
    #             torch.log(true_len + eps))**2).mean()
    # def angle_range_loss(
    #     lattice_pred,
    #     min_angle=60.0,
    #     max_angle=120.0,
    #     margin=5.0,
    #     edge_weight=0.1,
    # ):

    #     angles = lattice_pred[:, 3:]

    #     # 出界惩罚
    #     lower = F.relu(min_angle - angles)
    #     upper = F.relu(angles - max_angle)
    #     hard_loss = lower**2 + upper**2

    #     # 贴边排斥
    #     left_push  = F.relu((min_angle + margin) - angles)
    #     right_push = F.relu(angles - (max_angle - margin))

    #     edge_loss = (left_push/margin)**2 + (right_push/margin)**2

    #     return (hard_loss + edge_weight*edge_loss).mean()
    
    # def volume_loss(pred, target, log_tol=0.3, min_volume=5.0):

    #     pred_vol = lattice_volume(pred)
    #     true_vol = lattice_volume(target)

    #     log_diff = torch.log(pred_vol+1e-8) - torch.log(true_vol+1e-8)

    #     band = F.relu(torch.abs(log_diff) - log_tol)
    #     band_loss = band**2

    #     collapse = F.relu(min_volume - pred_vol)
    #     collapse_loss = collapse**2

    #     return band_loss.mean() + collapse_loss.mean()
    
    # loss_prop = torch.tensor(0.0, device=device)

    # # 原始分类 / 回归
    # for i in range(6):
    #     loss_prop += criterion(prop_preds[:, i], prop[:, i])

    # loss_prop = loss_prop / 6

    # # 物理约束
    # loss_phys = (
    #     1.0 * lattice_log_loss(prop_preds, prop)
    #     + 1.0 * volume_loss(prop_preds, prop)
    #     + 0.5 * angle_range_loss(prop_preds)
    # )

    # loss_prop = loss_prop*0.01 + loss_phys
    
    

    # ==============================
    # RMS metric (保持连续空间)
    # ==============================
    rms_losses = []
    correct = 0
    total = 0

    for i in range(B):

        Li = lengths[i].item()
        if Li <= 1:
            continue

        coord_gt = coords[i, :Li]       # fractional GT
        coord_pr = coord_preds[i, :Li]  # fractional prediction

        # -------------------------------------------------
        # 1️⃣ Remove global translation (important!)
        # -------------------------------------------------
        diff = coord_pr - coord_gt
        diff = diff - torch.round(diff)     # PBC wrap first

        shift = diff.mean(dim=0, keepdim=True)
        diff = diff - shift
        diff = diff - torch.round(diff)     # wrap again after shift

        # -------------------------------------------------
        # 2️⃣ Fractional RMSD (NO lattice / NO volume)
        # -------------------------------------------------
        rms = torch.sqrt((diff ** 2).sum(dim=1).mean())

        rms_losses.append(rms)

        correct += (rms < rms_threshold).float().item()
        total += 1

    # -------------------------------------------------
    # 3️⃣ Batch reduction
    # -------------------------------------------------
    rms_loss = torch.stack(rms_losses).mean() if rms_losses else torch.tensor(0.0, device=device)
    accuracy = correct / max(total, 1)

    return loss_coord, loss_prop * prop_weight, rms_loss, accuracy

def prop_preds_to_lattice_matrix(prop_preds):#使用分类方法进行的调制
    """
    Convert model prop predictions to lattice matrix.

    Supports THREE cases automatically:
        1) classification logits  : [B,6,n_bins]
        2) flattened logits       : [B,6*n_bins]
        3) continuous regression  : [B,6]

    Returns:
        L : [B,3,3] lattice matrix
    """

    B = prop_preds.shape[0]

    # ==============================
    # CASE 1 — logits [B,6,n_bins]
    # ==============================
    if prop_preds.dim() == 3:
        prop_ids = prop_preds.argmax(-1)

    # ==============================
    # CASE 2 — flattened logits [B,6*n_bins]
    # ==============================
    elif prop_preds.dim() == 2 and prop_preds.shape[1] > 6:
        n_bins = discretizer.n_bins
        prop_preds = prop_preds.view(B, 6, n_bins)
        prop_ids = prop_preds.argmax(-1)

    # ==============================
    # CASE 3 — already continuous [B,6]
    # ==============================
    elif prop_preds.shape[1] == 6:
        lat6 = prop_preds
        return lattice6_to_matrix(lat6)

    else:
        raise ValueError("Unknown prop_preds shape")

    # ==============================
    # decode bins → continuous lattice6
    # ==============================
    prop_ids = torch.clamp(prop_ids, 0, discretizer.n_bins - 1)
    lat6 = discretizer.decode(prop_ids)

    # ==============================
    # convert degrees → radians
    # ==============================
    lat6[:, 3:] = torch.deg2rad(lat6[:, 3:])

    # ==============================
    # build lattice matrix
    # ==============================
    L = lattice6_to_matrix(lat6)

    return L

'''def autoreg_loss(
    coord_preds, prop_preds,
    Zs, coords, prop, lengths, 
    rms_threshold=0.5, prop_weight=1
):
    """
    Coordinate autoregressive loss:
    - Z is condition only
    - coords[t] is predicted at position t
    - causality enforced by attention mask in forward
    """

    B, L, _ = coord_preds.shape
    device = coord_preds.device

    # -----------------------------
    # valid mask based on lengths
    # -----------------------------
    valid_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    for i in range(B):
        Li = lengths[i].item()
        valid_mask[i, :Li] = 1

    valid_flat = valid_mask.reshape(-1).float()

    # ==============================
    # coordinate loss (correct)
    # ==============================

    losses = []

    for i in range(B):

        Li = lengths[i].item()
        if Li <= 1:
            continue

        pred = coord_preds[i, :Li]
        tgt  = coords[i, :Li]
        
        shift = (pred - tgt).mean(dim=0, keepdim=True)
        pred = pred - shift

        # periodic minimal image
        diff = pred - tgt
        diff = diff - torch.round(diff)

        # Euclidean distance per atom
        dist = torch.sqrt(
            diff.pow(2).sum(dim=-1) + 1e-12
        )

        # structure-wise mean
        losses.append(dist.mean())

    loss_coord = torch.stack(losses).mean()

    if prop_preds is not None:
        if prop_preds.shape[-1] == 9:
            pred_lat = prop_preds.view(-1,3,3)
            true_lat = prop.view(-1,3,3)

            # ===== metric tensor loss =====
            Vp = torch.clamp(torch.abs(torch.det(pred_lat)), min=1e-6)
            Vt = torch.clamp(torch.abs(torch.det(true_lat)), min=1e-6)

            scale = Vt.pow(2/3).view(-1,1,1)

            Gp = (pred_lat.transpose(1,2) @ pred_lat) / scale
            Gt = (true_lat.transpose(1,2) @ true_lat) / scale

            loss_metric = ((Gp - Gt)**2).mean()

            # ===== volume loss =====
            vol_pred = torch.clamp(torch.abs(torch.det(pred_lat)), min=1e-6)
            vol_true = torch.clamp(torch.abs(torch.det(true_lat)), min=1e-6)

            loss_vol = (torch.log(vol_pred) - torch.log(vol_true)).pow(2).mean()

            loss_prop = loss_metric + 0.1 * loss_vol

        elif prop_preds.shape[-1] == 6:
        # -----------------------------
        # property regression loss
        # -----------------------------
            eps = 1e-6
            # 对长度类属性用 softplus 保证非负
            pred_len = F.softplus(prop_preds[:, :3])
            true_len = prop[:, :3].clamp(min=eps)

            loss_len = (torch.log(pred_len + eps) - torch.log(true_len + eps)).pow(2).mean()
            loss_ang = (prop_preds[:, 3:] - prop[:, 3:]).pow(2).mean()
            loss_prop = loss_len + loss_ang
    else:loss_prop = 0

    rms_losses = []
    correct = 0
    total = 0

    for i in range(B):

        Li = lengths[i].item()
        if Li <= 1:
            continue

        coord_gt = coords[i, :Li]
        coord_pr = coord_preds[i, :Li]

        # remove global shift
        shift = (coord_pr - coord_gt).mean(dim=0, keepdim=True)
        coord_pr = coord_pr - shift

        # periodic minimal image
        diff = coord_pr - coord_gt
        diff = diff - torch.round(diff)

        # ===== NEW: 6 -> lattice matrix =====
        lat6 = prop[i]   # (6,)
        L = lattice6_to_matrix(lat6)

        G = L.t() @ L

        # real-space RMS
        d2 = torch.einsum('ni,ij,nj->n', diff, G, diff)
        rms_real = torch.sqrt(d2.mean())

        # normalization
        V = torch.clamp(torch.abs(torch.det(L)), min=1e-6)
        norm = (V / Li).pow(1/3)

        rms = rms_real / norm
        rms_losses.append(rms)

        with torch.no_grad():
            correct += (rms < rms_threshold).float().item()
            total += 1

        rms_loss = torch.stack(rms_losses).mean() if rms_losses else torch.tensor(0.0, device=device)
        accuracy = correct / max(total, 1)

    return loss_coord, loss_prop * prop_weight, rms_loss, accuracy'''

# =========================================================
#  LATTICE MATCHER-STYLE LOSS
# =========================================================
def lattice_match_loss(pred_lat, true_lat):
    """
    Matcher-style lattice loss with full stability.

    pred_lat, true_lat: (B,3,3)
    returns scalar loss
    """
    eps = 1e-6

    # -------------------------
    # Volume normalization
    # -------------------------
    V_pred = torch.clamp(torch.abs(torch.det(pred_lat)), min=eps)
    V_true = torch.clamp(torch.abs(torch.det(true_lat)), min=eps)

    scale_pred = V_pred.pow(1/3).view(-1,1,1)
    scale_true = V_true.pow(1/3).view(-1,1,1)

    pred_n = pred_lat / scale_pred
    true_n = true_lat / scale_true

    # -------------------------
    # Kabsch alignment (fully differentiable)
    # -------------------------
    H = pred_n.transpose(1,2) @ true_n

    # small jitter for numerical stability
    H = H + 1e-6 * torch.eye(3, device=H.device).unsqueeze(0)

    U, S, Vt = torch.linalg.svd(H)

    # reflection correction
    det = torch.det(Vt.transpose(1,2) @ U.transpose(1,2))
    D = torch.diag_embed(torch.stack([
        torch.ones_like(det),
        torch.ones_like(det),
        det
    ], dim=-1))

    R = Vt.transpose(1,2) @ D @ U.transpose(1,2)

    pred_align = pred_n @ R

    # -------------------------
    # RMS difference
    # -------------------------
    diff = pred_align - true_n
    rms = torch.sqrt((diff**2).sum(dim=(1,2)) / 9.0)

    return rms.mean()


# =========================================================
#  METRIC TENSOR LOSS (ultra stable regularizer)
# =========================================================
def lattice_metric_loss(pred_lat, true_lat):
    Gp = pred_lat.transpose(1,2) @ pred_lat
    Gt = true_lat.transpose(1,2) @ true_lat
    return ((Gp - Gt)**2).mean()


# =========================================================
#  MAIN AUTOREGRESSIVE LOSS
# =========================================================
'''def autoreg_loss(
    coord_preds, prop_preds,
    Zs, coords, prop, lengths,
    rms_threshold=0.1,
    prop_weight=0.1
):

    B, L, _ = coord_preds.shape
    device = coord_preds.device

    # ==============================
    # mask
    # ==============================
    valid_mask = torch.zeros((B, L), device=device)
    for i in range(B):
        valid_mask[i, :lengths[i]] = 1

    # ==============================
    # PERIODIC COORD LOSS
    # ==============================
    diff = coord_preds - coords
    diff = diff - torch.round(diff)   # minimal image

    sq = (diff**2).sum(-1)
    loss_coord = (sq * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    # ==============================
    # LATTICE LOSS (STABLE VERSION)
    # ==============================
    loss_prop = torch.tensor(0.0, device=device)

    if prop_preds is not None and prop_preds.shape[-1] == 9:

        pred_lat = prop_preds.view(-1,3,3)
        true_lat = prop.view(-1,3,3)

        # metric tensor loss
        Gp = pred_lat.transpose(1,2) @ pred_lat
        Gt = true_lat.transpose(1,2) @ true_lat

        loss_metric = ((Gp - Gt)**2).mean()

        # volume loss
        vol_pred = torch.clamp(torch.abs(torch.det(pred_lat)), min=1e-6)
        vol_true = torch.clamp(torch.abs(torch.det(true_lat)), min=1e-6)

        loss_vol = (torch.log(vol_pred) - torch.log(vol_true)).pow(2).mean()

        loss_prop = loss_metric + 0.1 * loss_vol

    # ==============================
    # RMS accuracy (no gradient)
    # ==============================
    with torch.no_grad():
        rms_list = []
        correct = 0
        total = 0

        for i in range(B):
            Li = lengths[i]
            if Li <= 1:
                continue

            diff = coord_preds[i,:Li] - coords[i,:Li]
            diff = diff - torch.round(diff)

            lat = prop[i].view(3,3)
            V = torch.clamp(torch.abs(torch.det(lat)), min=1e-6)
            norm = (V / Li).pow(1/3)

            rms = torch.sqrt((diff**2).sum(-1).mean()) / norm
            rms_list.append(rms)

            correct += (rms < rms_threshold).item()
            total += 1

        rms_loss = torch.stack(rms_list).mean() if rms_list else torch.tensor(0.0)
        accuracy = correct / max(total,1)

    return loss_coord, loss_prop * prop_weight, rms_loss, accuracy'''

def frac_to_cart(coords_frac, lattice6):
    """
    coords_frac: [B,L,3]
    lattice6: [B,6]
    """
    L = lattice6_to_matrix(lattice6)  # [B,3,3]
    coords_cart = torch.matmul(coords_frac, L)  # [B,L,3]
    return coords_cart