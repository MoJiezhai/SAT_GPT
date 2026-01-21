import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from SAT import SATGPT
from torch.utils.data import DataLoader
from data import SATDataLoader, collate_fn, SequenceStructureDataset, CrystalStructureDataset, parse_atoms, data_add
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import argparse
import os

def lattice_volume_non_ortho(prop):
    """
    prop: (6,) tensor -> [a, b, c, alpha, beta, gamma]
    angles in degree
    """
    a, b, c = prop[:3]
    alpha, beta, gamma = prop[3:] * torch.pi / 180.0  # deg -> rad

    cos_a = torch.cos(alpha)
    cos_b = torch.cos(beta)
    cos_c = torch.cos(gamma)

    volume = a * b * c * torch.sqrt(
        1
        + 2 * cos_a * cos_b * cos_c
        - cos_a**2
        - cos_b**2
        - cos_c**2
    ).clamp(min=1e-6)  # 数值稳定

    return volume

def autoreg_loss(
    coord_preds, prop_preds,
    Zs, coords, prop, lengths, 
    rms_threshold=0.1, prop_weight=0.1
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

    # -----------------------------
    # coordinate MSE loss
    # -----------------------------
    pred_flat = coord_preds.reshape(-1, 3)
    tgt_flat  = coords.reshape(-1, 3)

    # 数值安全处理，防止 NaN
    pred_flat = torch.nan_to_num(pred_flat, nan=0.0, posinf=0.0, neginf=0.0)
    tgt_flat  = torch.nan_to_num(tgt_flat, nan=0.0, posinf=0.0, neginf=0.0)

    loss_coord_all = (pred_flat - tgt_flat).pow(2).sum(dim=-1)
    loss_coord = (loss_coord_all * valid_flat).sum() / valid_flat.sum().clamp(min=1.0)

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

    # -----------------------------
    # structure-level RMS accuracy
    # -----------------------------
    rms_losses = []
    correct = 0
    total = 0

    for i in range(B):
        Li = lengths[i].item()

        coord_gt = coords[i, :Li]
        coord_pr = coord_preds[i, :Li]

        if coord_gt.shape[0] <= 1:
            continue

        N = coord_gt.shape[0]
        V = lattice_volume_non_ortho(prop[i])
        V = torch.clamp(V, min=1e-6)  # 防止负体积导致 NaN
        norm = (V / N).pow(1.0 / 3.0)

        rms = torch.sqrt((coord_pr - coord_gt).pow(2).sum(dim=-1).mean()) / norm
        rms_losses.append(rms)

        with torch.no_grad():
            correct += (rms < rms_threshold).float().item()
            total += 1

    rms_loss = torch.stack(rms_losses).mean() if rms_losses else torch.tensor(0.0, device=device)
    accuracy = correct / max(total, 1)

    return loss_coord * 0.1 + rms_loss, loss_prop * prop_weight, loss_coord, accuracy




# -----------------------
# Autoregressive generation
# -----------------------
@torch.no_grad()
def generate_autoregressive(model, initial_Z, initial_coords, max_new_atoms=20, device='cpu', temperature=1.0):
    """
    initial_Z: [L] int tensor
    initial_coords: [L,3] float tensor
    We'll iteratively predict next Z and then next coords, append and continue.
    """
    model.eval()
    Zs = initial_Z.clone().to(device).unsqueeze(0)  # [1,L]
    coords = initial_coords.clone().to(device).unsqueeze(0)  # [1,L,3]
    L = Zs.shape[1]
    for step in range(max_new_atoms):
        lengths = torch.tensor([Zs.shape[1]], dtype=torch.long, device=device)
        type_logits, coord_preds, _ = model(Zs, coords, lengths)
        # get last position logits (for predicting next token from last context position)
        # type_logits: [1, L, num_types], we want logits at pos L-1 -> will predict Z_{L}
        last_logits = type_logits[0, -1]  # [num_types]
        # sample or argmax
        probs = F.softmax(last_logits / temperature, dim=-1)
        next_Z = torch.multinomial(probs, num_samples=1).squeeze().unsqueeze(0)  # shape [1]
        # Append predicted Z to the sequence. For coords: we need to predict next coords based on appended Z.
        # To do that, we will append a placeholder coordinate (e.g., zeros) and run model again to predict coords at new last position,
        # alternatively we could let model directly output coords for next position using same last hidden vector. Here we take coord_preds at last pos
        # as prediction for next coordinate (shift logic consistent with training).
        next_coord = coord_preds[0, -1].unsqueeze(0).unsqueeze(0)  # [1,1,3]
        # append
        Zs = torch.cat([Zs, next_Z.unsqueeze(0)], dim=1)  # now [1, L+1]
        coords = torch.cat([coords, next_coord], dim=1)   # [1, L+1,3]
        # stop condition optionally when predict a special STOP token (if defined)
    return Zs.squeeze(0).cpu(), coords.squeeze(0).cpu()

# -----------------------
# Example usage (toy)
# -----------------------
atom_types = 118#可用的原子种类
from timeit import default_timer as timer
def toy_example(satopt, train_loader, val_loader, test_loader, ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_file = open("try_log.txt", "w")
    
    # model
    model = SATGPT(num_atom_types=satopt["model"]["num_atom_types"], y_dim=satopt["model"]["y_dim"], d_model=satopt["model"]["d_model"], n_layer=satopt["model"]["n_layer"], n_head=satopt["model"]["n_head"], d_head=satopt["model"]["d_head"], d_ff=satopt["model"]["d_ff"], coord_emb_dim=satopt["model"]["coord_emb_dim"], max_length=satopt["model"]["max_length"])
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=satopt["training"]["optamizer"]["lr"],
        weight_decay=satopt["training"]["optamizer"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=satopt["training"]["epoch"],
        eta_min=1e-7
    )
    
    def run_epoch(
        model,
        dataloader,
        device,
        epoch,
        phase="train",   # "train" | "val" | "test"
        coord_scale=1.0,
        prop_scale=1.0,
        optimizer=None,
        scheduler=None,
        save_traj=False,
        traj_path=None,
        batch_size=None,
        log_file=None,
        best_val=0
    ):
        start_time = timer()
        assert phase in ["train", "val", "test"]

        is_train = (phase == "train")

        if is_train:
            model.train()
        else:
            model.eval()

        with torch.no_grad():
            total_loss = 0.0
            total_type = 0.0
            total_coord = 0.0
            total_prop = 0.0
            total_rms = 0.0
            total_acc = 0.0

        step = 0

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for batch in dataloader:
                Z = batch['Z'].to(device)
                coords = batch['coords'].to(device)
                lengths = batch['lengths'].to(device)
                prop = batch['prop'].to(device)

                if phase == "train":
                    coord_preds, prop_preds = model(
                        Z,
                        coords,
                        lengths,
                        prop,
                        None,
                        batch['sub_nodes'].to(device),
                        batch['sub_indicator'].to(device),
                        batch['sub_batch_index'].to(device),
                        False
                    )
                else:
                    coord_preds = model.sample(Z, cond=None)
                    _, prop_preds = model(
                        Z,
                        coord_preds,
                        lengths,
                        None,
                        None,
                        None,
                        None,
                        None,
                        False
                    )  # 或者用生成结构再算

                loss_coord, loss_prop, rms_loss, accuracy = autoreg_loss(
                    coord_preds,
                    prop_preds,
                    Z,
                    coords,
                    prop,
                    lengths
                )
                

                loss = prop_scale * loss_prop + coord_scale * loss_coord 

                if is_train:
                    #c = timer()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    #print(timer()-c)

                with torch.no_grad():
                    total_loss += loss.item()
                    total_coord += loss_coord.item()
                    total_prop += loss_prop.item()
                    total_rms += rms_loss.item()
                    total_acc += accuracy

                # -------- 保存轨迹（通常只在 test）--------
                if save_traj:
                    from match import preds_to_traj
                    traj_path_step = traj_path+f"{step:03d}.traj"
                    preds_to_traj(
                        type_logits=batch['Z'].to(device),
                        coord_preds=coord_preds,
                        lengths=batch['lengths'].to(device),
                        traj_path=traj_path_step,
                        cell_params=prop_preds,
                        ignore_index_list=batch['score']
                    )
                    step += 1

        if scheduler is not None and is_train:
            scheduler.step()

        with torch.no_grad():
            n = len(dataloader)
            avg_loss = total_loss / n
            avg_type = total_type / n
            avg_coord = total_coord / n
            avg_prop = total_prop / n
            avg_rms = total_rms / n
            avg_acc = total_acc / n
        
        end_time = timer()
        time = end_time-start_time

        # ------------------ 输出 ------------------
        msg = (
            f"[Epoch {epoch}] [{phase.upper()}] "
            f"Total {avg_loss:.4f} | "
            f"Type {avg_type:.4f} | "
            f"Coord+Rms {avg_coord:.4f} | "
            f"Prop {avg_prop:.4f} | "
            f"Coord {avg_rms:.4f} | "
            f"Acc {avg_acc:.4f} | "
            f"Time {time:.4f}"
        )
        print(msg)

        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

        if phase == 'val' and avg_loss < best_val :
            best_val = avg_loss
            torch.save(model.state_dict(), "best.pt")
            print("Saved best model.")
            log_file.write("Saved best model.\n")
        return best_val

    # train few steps
    best_val = float('inf')
    coord_scale = 1
    prop_scale = 1
    start_epoch = 0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        if scheduler is not None and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])

        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]

        print(f"Resumed from epoch {start_epoch}, best_val = {best_val:.4f}")
    for epoch in range(start_epoch,satopt["training"]["epoch"]):
        
        # -------- TRAIN --------
        run_epoch(
            model,
            train_loader,
            device,
            epoch,
            phase="train",
            coord_scale=coord_scale,
            prop_scale=prop_scale,
            optimizer=optimizer,
            scheduler=scheduler,
            log_file=log_file
        )
        # -------- VAL --------
        best_val = run_epoch(
            model,
            val_loader,
            device,
            epoch,
            phase="val",
            coord_scale=coord_scale,
            prop_scale=prop_scale,
            save_traj=False,
            traj_path="SAT_data/gen/val_gen",
            batch_size=batch_size,
            log_file=log_file,
            best_val=best_val
        )
        print(best_val)

    # generation demo
    model.load_state_dict(torch.load("best.pt"))

    # -------- test --------  
    run_epoch(
        model,
        test_loader,
        device,
        epoch,
        phase="test",
        coord_scale=coord_scale,
        prop_scale=prop_scale,
        save_traj=True,
        traj_path="SAT_data/gen/gen",
        log_file=log_file
    )

    log_file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_path = args.train_json
    val_path = args.val_json
    test_path = args.test_json
    train_ds = CrystalStructureDataset(train_path, randomchose=True,dim=2, scale=0.001)
    val_ds   = CrystalStructureDataset(val_path,   randomchose=False,dim=2, scale=0.001)
    test_ds   = CrystalStructureDataset(test_path,   randomchose=False,dim=2, scale=0.001)
    import random
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    from match import dataset_to_traj
    dataset_to_traj(
    test_ds,
    traj_filename="SAT_data/target/target.traj"
    )
    print('生成完成')
    
    # 完整周期表：符号 → 原子序数
    
    train_dataset = SequenceStructureDataset(train_ds, max_atoms=70)#Dataset
    val_dataset = SequenceStructureDataset(val_ds, max_atoms=70)#Dataset
    test_dataset = SequenceStructureDataset(test_ds, max_atoms=70)#Dataset
    
    batch_size=args.batch_size
    train_loader = DataLoader(
        SATDataLoader(train_dataset, cutoff=0.5, k_hop=1),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        SATDataLoader(val_dataset, cutoff=0.5, k_hop=1),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SATDataLoader(test_dataset, cutoff=0.5, k_hop=1),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    satopt = {
        "model": {
            "num_atom_types":atom_types,
            "y_dim":6, 
            "d_model":128, 
            "n_layer":4, 
            "n_head":4, 
            "d_head":32,
            "d_ff":512, 
            "coord_emb_dim":256, 
            "max_length":32
        },
        "training": {
            "epoch": args.epochs,
            "optamizer": {
                "lr": 1e-5,
                "weight_decay": 0.0
            }
        }
    }
    
    toy_example(satopt, train_loader, val_loader, test_loader)