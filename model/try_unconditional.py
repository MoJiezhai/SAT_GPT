import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from SAT_llm import SATGPT
from torch.utils.data import DataLoader
from data import SATDataLoader, collate_fn, SequenceStructureDataset, CrystalStructureDataset
import argparse
import os
from model import Transformer
from Z_model import CompositionPredictor, CompositionDecoder
from loss import autoreg_loss, lattice6_to_matrix

def save_checkpoint(path, epoch, model, prop_model, Z_model,
                    optimizer, prop_optimizer, Z_optimizer,
                    scheduler, prop_scheduler, Z_scheduler,
                    best_val):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "prop_model_state": prop_model.state_dict(),
        "Z_model_state": Z_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "prop_optimizer_state": prop_optimizer.state_dict(),
        "Z_optimizer_state": Z_optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "prop_scheduler_state": prop_scheduler.state_dict() if prop_scheduler else None,
        "Z_scheduler_state": Z_scheduler.state_dict() if Z_scheduler else None,
        "best_val": best_val,
    }
    torch.save(ckpt, path)

# -----------------------
# Example usage (toy)
# -----------------------
atom_types = 118#可用的原子种类
from timeit import default_timer as timer
def toy_example(satopt, train_loader, val_loader, test_loader, ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_file = open("try_log.txt", "w")
    
    prop_model = Transformer(num_cond=satopt["model"]["k_dim"],max_atoms=satopt["model"]["max_length"])
    prop_model.to(device)
    prop_optimizer = torch.optim.AdamW(prop_model.parameters(), lr=satopt["training"]["optimizer"]["lr"], weight_decay=1e-5)
    prop_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    prop_optimizer,
    T_max=satopt["training"]["epoch"],
    eta_min=1e-7
)
    Z_model = CompositionDecoder(num_cond=satopt["model"]["k_dim"],max_atoms=satopt["model"]["max_length"]//4)
    Z_model.to(device)
    Z_optimizer = torch.optim.AdamW(Z_model.parameters(), lr=satopt["training"]["optimizer"]["lr"], weight_decay=1e-5)
    Z_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    Z_optimizer,
    T_max=satopt["training"]["epoch"],
    eta_min=1e-7
)
    # model
    model = SATGPT(num_atom_types=satopt["model"]["num_atom_types"], k_dim=satopt["model"]["k_dim"], d_model=satopt["model"]["d_model"], n_layer=satopt["model"]["n_layer"], n_head=satopt["model"]["n_head"], d_head=satopt["model"]["d_head"], d_ff=satopt["model"]["d_ff"], num_bins=100, max_length=satopt["model"]["max_length"])
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=satopt["training"]["optimizer"]["lr"]*1,
        weight_decay=satopt["training"]["optimizer"]["weight_decay"]
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
        total_scale=1.0,
        optimizer=None,
        scheduler=None,
        save_traj=False,
        traj_path=None,
        batch_size=None,
        log_file=None,
        best_val=0,
    ):
        start_time = timer()
        assert phase in ["train", "val", "test"]

        is_train = (phase == "train")

        if is_train:
            model.train()
        else:
            model.eval()
            
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
                BOS_Z = atom_types+1
                EOS_Z = atom_types+2
                coords_frac  = batch['coords'].to(device)
                from data import apply_random_shift
                
                coords_frac_add, _ = apply_random_shift(coords_frac, 0.3)#*(satopt["training"]["epoch"]-epoch)/satopt["training"]["epoch"])#min((epoch-satopt["training"]["epoch"]/3)/satopt["training"]["epoch"]*2,1))
                coords_frac = coords_frac_add
                lengths = batch['lengths'].to(device)
                prop = batch['prop'].to(device)
                cond = batch['cond'].to(device)
                B, L, _ = coords_frac.shape
                
                def build_structure_sequence(coords, Z,  EOS_Z):
                    B, L = Z.shape
                    device = Z.device
                    Z_repeat = (
                        Z.unsqueeze(-1)
                        .repeat(1, 1, 3)
                        .reshape(B, -1)
                    )
                    Z_seq = torch.full(
                        (B, 3 * L + 1),
                        EOS_Z,
                        device=device,
                        dtype=Z.dtype
                    )
                    Z_seq[:, 1:] = Z_repeat
                    B, L, _ = coords.shape
                    coord_seq = coords.reshape(B, -1)
                    coord_seq_bos = torch.zeros(
                        (B, 3 * L + 1),
                        device=device,
                        dtype=coords.dtype
                    )
                    coord_seq_bos[:, 1:] = coord_seq
                    lengths_seq = lengths * 3 + 1
                    return Z_seq, coord_seq_bos, lengths_seq
                
                def build_composition_sequence(
                    Z,
                    BOS_Z,
                    EOS_Z,
                    PAD_Z=0
                ):
                   

                    B, L = Z.shape
                    device = Z.device

                    seq_len = L + 2

                    Z_seq = torch.full(
                        (B, seq_len),
                        PAD_Z,
                        device=device,
                        dtype=Z.dtype
                    )

                    Z_seq[:, 0] = BOS_Z
                    Z_seq[:, 1:L+1] = Z

                    eos_pos = lengths + 1

                    batch_idx = torch.arange(
                        B,
                        device=device
                    )

                    Z_seq[
                        batch_idx,
                        eos_pos
                    ] = EOS_Z

                    pad_mask = (
                        torch.arange(seq_len, device=device)
                        .unsqueeze(0)
                        > eos_pos.unsqueeze(1)
                    )

                    Z_seq[pad_mask] = PAD_Z

                    return Z_seq
                
                def build_Z_sequence(
                    Z,
                    BOS_Z,
                    EOS_Z,
                    PAD_Z=0
                ):
                    
                    B, T = Z.shape
                    device = Z.device

                    sequences = []
                    lengths = []

                    max_len = 0

                    for b in range(B):

                        seq = Z[b]

                        # ---------------------------------
                        # remove PAD
                        # ---------------------------------
                        valid = seq[seq != PAD_Z]

                        if len(valid) == 0:

                            parsed = torch.empty(
                                0,
                                device=device,
                                dtype=Z.dtype
                            )

                            atom_len = 0

                        else:

                            bos_pos = (
                                valid == BOS_Z
                            ).nonzero(as_tuple=True)[0]

                            if len(bos_pos) > 0:

                                start = bos_pos[0].item() + 1

                            else:

                                start = 0

                            valid = valid[start:]

                            eos_pos = (
                                valid == EOS_Z
                            ).nonzero(as_tuple=True)[0]

                            if len(eos_pos) > 0:

                                end = eos_pos[0].item()

                                parsed = valid[:end]

                            else:

                                parsed = valid

                            parsed = parsed[
                                (parsed != BOS_Z)
                                & (parsed != EOS_Z)
                            ]

                            atom_len = len(parsed)

                        sequences.append(parsed)

                        lengths.append(atom_len)

                        max_len = max(max_len, atom_len)

                    Z_gen = torch.full(
                        (B, max_len),
                        PAD_Z,
                        device=device,
                        dtype=Z.dtype
                    )

                    for b, seq in enumerate(sequences):

                        if len(seq) > 0:

                            Z_gen[b, :len(seq)] = seq

                    lengths = torch.tensor(
                        lengths,
                        device=device,
                        dtype=torch.long
                    )

                    return Z_gen, lengths
                
                if phase == "train" :
                    out, loss_type = Z_model(
                        Z,
                        cond=cond,
                        return_loss=True
                    )
                    Z_seq, coord_seq_bos, lengths_seq = build_structure_sequence(coords=coords_frac,Z=Z,EOS_Z=BOS_Z)
                    prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths, temperature=0.05)  
                    logits, loss_coord = model(
                        Zs=Z_seq, 
                        coords=coord_seq_bos,
                        lengths=lengths_seq, 
                        lattice=prop,
                        cond=cond,
                        mode='train'
                    )
                    pred_seq = torch.argmax(logits, dim=-1) 
                    pred_seq = pred_seq[:, 1:]
                    coord_preds = pred_seq.reshape(B, L, 3).float() / model.num_bins
                    
                elif phase == "val":
                    Z_seq, coord_seq_bos, lengths_seq = build_structure_sequence(coords=coords_frac,Z=Z,EOS_Z=BOS_Z)
                    Z_gen = Z_model.generate(cond=cond)
                    
                    prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths, temperature=0.05)  
                    coord_preds = model.generate(
                        Zs=Z_seq, 
                        lengths=lengths_seq, 
                        lattice=prop_preds, 
                        cond=cond)
                    pred_seq = coord_preds[:, 1:]
                    coord_preds = pred_seq.reshape(B, L, 3).float() 
                    
                else:
                    Z_gen = Z_model.generate(cond=cond)
                    Z_gen, lengths = build_Z_sequence(Z_gen, BOS_Z, EOS_Z)
                    Z_seq, _, _ = build_structure_sequence(coords=coords_frac,Z=Z_gen,EOS_Z=BOS_Z)
                    prop_preds = prop_model(Z=Z_gen, cond=cond, lengths=lengths, temperature=0.05) 
                    coord_preds = model.generate(
                        Zs=Z_seq, 
                        lengths=lengths*3+1, 
                        lattice=prop_preds,
                        cond=cond)
                    pred_seq = coord_preds[:, 1:]
                    _, L = Z_gen.shape
                    coord_preds = pred_seq.reshape(B, L, 3).float() 
                
                    
                
                if phase == 'val': 
                    loss_type_new, loss_coord_new, loss_prop, rms_loss, accuracy = autoreg_loss(
                        None,
                        coord_preds,
                        prop_preds,
                        Z,
                        coords_frac,
                        prop,
                        lengths,
                        100
                    )
                    
                    loss_coord=loss_coord_new
                    loss_type=loss_type_new
                elif phase == 'train':
                    _, _, loss_prop, rms_loss, accuracy = autoreg_loss(
                        None,
                        None,
                        prop_preds,
                        Z,
                        coords_frac,
                        prop,
                        lengths,
                        100
                    ) 
                    valid_mask = torch.zeros_like(coords_frac[..., 0])  
                    for i in range(B):
                        valid_mask[i, :lengths[i]] = 1
                    valid_mask = valid_mask.unsqueeze(-1) 
                    pred = coord_preds  
                    target = coords_frac 
                    valid_mask = valid_mask
                    loss = ((pred - target) ** 2) *  valid_mask
                    denom = ( valid_mask).sum().clamp(min=1)
                    loss_coord_new = loss.sum() / denom
                    loss_coord += loss_coord_new * (1 + epoch / satopt['training']['epoch'])
                else:
                    loss_coord = torch.tensor(0.0, device=device)
                    loss_type = torch.tensor(0.0, device=device)
                    loss_prop = torch.tensor(0.0, device=device)
                    rms_loss = torch.tensor(0.0, device=device)
                    accuracy = torch.tensor(0.0, device=device)

                
                loss = loss_type + prop_scale * loss_prop + coord_scale * loss_coord 

                if is_train:
                    optimizer.zero_grad()
                    prop_optimizer.zero_grad()
                    Z_optimizer.zero_grad()
                    loss_prop.backward()
                    loss_coord.backward()
                    loss_type.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    torch.nn.utils.clip_grad_norm_(prop_model.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(Z_model.parameters(), 1.0)
                    optimizer.step()
                    prop_optimizer.step()
                    Z_optimizer.step()

                
                with torch.no_grad():
                    total_loss += loss.item()
                    total_type += loss_type.item()
                    total_coord += loss_coord.item()
                    total_prop += loss_prop
                    total_rms += rms_loss.item()
                    total_acc += accuracy


                if save_traj:
                    from match import preds_to_traj
                    traj_path_step = traj_path+f"{step:03d}.traj"
                    L = lattice6_to_matrix(prop_preds)
                    cart = torch.matmul(coord_preds,L)
                    preds_to_traj(
                        type_logits=batch['Z'].to(device),
                        coord_preds=cart,
                        lengths=batch['lengths'].to(device),
                        traj_path=traj_path_step,
                        cell_params=L,
                        ignore_index_list=batch['score']
                    )
                step += 1

        if scheduler is not None and is_train:
            scheduler.step()
            prop_scheduler.step()
            Z_scheduler.step()

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

        msg = (
            f"[Epoch {epoch}] [{phase.upper()}] "
            f"Total {avg_loss:.4f} | "
            f"Type {avg_type:.4f} | "
            f"Coord {avg_coord:.4f} | "
            f"Prop {avg_prop:.4f} | "
            f"Rms {avg_rms:.4f} | "
            f"Acc {avg_acc:.4f} | "
            f"Time {time:.4f}"
        )
        print(msg)

        if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

        if phase == 'val' and avg_loss < best_val:
            best_val = avg_loss
            save_checkpoint(
                traj_path + "/best_ckpt.pt",
                epoch,
                model,
                prop_model,
                Z_model,
                optimizer,
                prop_optimizer,
                Z_optimizer,
                scheduler,
                prop_scheduler,
                Z_scheduler,
                best_val
            )
            print("Saved best model.")
            log_file.write("Saved best model.\n")
        elif phase == 'train' and traj_path is not None:
            save_checkpoint(
                traj_path + "/last_ckpt.pt",
                epoch,
                model,
                prop_model,
                Z_model,
                optimizer,
                prop_optimizer,
                Z_optimizer,
                scheduler,
                prop_scheduler,
                Z_scheduler,
                best_val
            )
            print("Saved last model.")
            log_file.write("Saved last model.\n")
        return best_val

    best_val = float('inf')
    coord_scale = 1
    prop_scale = 1
    start_epoch = 0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        prop_model.load_state_dict(ckpt["prop_model_state"])
        Z_model.load_state_dict(ckpt["Z_model_state"])

        optimizer.load_state_dict(ckpt["optimizer_state"])
        prop_optimizer.load_state_dict(ckpt["prop_optimizer_state"])
        Z_optimizer.load_state_dict(ckpt["Z_optimizer_state"])

        if scheduler is not None and ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if prop_scheduler is not None and ckpt["prop_scheduler_state"] is not None:
            prop_scheduler.load_state_dict(ckpt["prop_scheduler_state"])
        if Z_scheduler is not None and ckpt["Z_scheduler_state"] is not None:
            Z_scheduler.load_state_dict(ckpt["Z_scheduler_state"])

        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]

        print(f"Resumed from epoch {start_epoch}, best_val = {best_val:.4f}")
    for epoch in range(start_epoch,satopt["training"]["epoch"]):
        
        run_epoch(
            model,
            train_loader,
            device,
            epoch,
            phase="train",
            coord_scale=coord_scale*(1+4*epoch/satopt['training']['epoch']),
            prop_scale=prop_scale,
            total_scale=1*epoch/satopt['training']['epoch'],
            optimizer=optimizer,
            scheduler=scheduler,
            log_file=log_file,
            batch_size=batch_size,
            traj_path=args.save_dir if epoch % 10 == 9 else None,
            best_val=best_val,
        )
        if epoch%40==1 or epoch == satopt["training"]["epoch"]:best_val = run_epoch(
            model,
            val_loader,
            device,
            epoch,
            phase="val",
            coord_scale=coord_scale,
            prop_scale=prop_scale,
            save_traj=False,
            traj_path=args.save_dir,
            batch_size=batch_size,
            log_file=log_file,
            best_val=best_val,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        
    run_epoch(
        model,
        test_loader,
        device,
        epoch=satopt["training"]["epoch"],
        phase="test",
        coord_scale=coord_scale,
        prop_scale=prop_scale,
        batch_size=batch_size,
        save_traj=True,
        traj_path=args.save_dir+"/gen/gen",
        log_file=log_file,
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
    train_ds = CrystalStructureDataset(train_path, randomchose=False,dim=2, scale=1)
    val_ds   = CrystalStructureDataset(val_path,   randomchose=False,dim=2, scale=1)
    test_ds   = CrystalStructureDataset(test_path,   randomchose=False,dim=2, scale=1)
    import random
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    from match import dataset_to_traj
    dataset_to_traj(
    train_ds,
    traj_filename=args.save_dir+"/target/train.traj"
    )
    dataset_to_traj(
    val_ds,
    traj_filename=args.save_dir+"/target/val.traj"
    )
    dataset_to_traj(
    test_ds,
    traj_filename=args.save_dir+"/target/target.traj"
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
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        SATDataLoader(val_dataset, cutoff=0.5, k_hop=1),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SATDataLoader(test_dataset, cutoff=0.5, k_hop=1),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    
    satopt = {
        "model": {
            "k_dim":1,
            "d_model": 512,        # ↓
            "n_layer": 12,         # ↓
            "n_head": 8,           # ↓
            "d_ff": 1024,          # ok
            "d_head":64,
            "coord_emb_dim": 64,   # ↓
            "max_length":80,
            "num_atom_types":118,
        },
        "training": {
            "epoch": args.epochs,
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 1e-3
            }
        }
    }
    
    toy_example(satopt, train_loader, val_loader, test_loader, 
    ckpt_path=os.path.join(args.save_dir, "last_ckpt.pt"))
