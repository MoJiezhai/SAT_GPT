import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from model import Transformer
from loss import autoreg_loss

from timeit import default_timer as timer

from data import (
    SATDataLoader,
    collate_fn,
    SequenceStructureDataset,
    CrystalStructureDataset
)

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
            if scheduler is not None else None,
            "best_val": best_val,
        },
        path
    )

def run_epoch(
    model,
    loader,
    device,
    epoch,
    phase="train",
    optimizer=None,
    scheduler=None,
    log_file = None
):
    start_time = timer()

    is_train = phase == "train"

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0

    context = (
        torch.enable_grad()
        if is_train
        else torch.no_grad()
    )

    with context:

        for batch in loader:

            Z = batch["Z"].to(device)
            lengths = batch["lengths"].to(device)

            cond = batch["cond"].to(device)

            lattice_gt = batch["prop"].to(device)

            coords_frac = batch["coords"].to(device)
            
            #print(cond)

            lattice_pred = model(
                Z=Z,
                cond=cond,
                lengths=lengths,
                temperature=0.05,
            )

            _, _, loss_prop, rms_loss, accuracy = autoreg_loss(
                None,
                None,
                lattice_pred,
                Z,
                coords_frac,
                lattice_gt,
                lengths,
                100
            )

            loss = loss_prop

            if is_train:

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                optimizer.step()

            total_loss += loss.item()

    if is_train and scheduler is not None:
        scheduler.step()

    n = len(loader)

    avg_loss = total_loss / n
    
    end_time = timer()
    time = end_time-start_time
    
    msg = (
        f"[Epoch {epoch}] [{phase.upper()}] "
        f"Loss={avg_loss:.6f} "
        f"Time={time:.2f}s"
    )

    print(msg)
    
    if log_file is not None:
            log_file.write(msg + "\n")
            log_file.flush()

    return avg_loss

def train_prop_model(
    satopt,
    train_loader,
    val_loader,
    test_loader,
    save_dir,
    ckpt_path=None,
):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = Transformer(
        num_cond=satopt["model"]["k_dim"],
        max_atoms=satopt["model"]["max_length"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=satopt["training"]["optimizer"]["lr"],
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=satopt["training"]["epoch"],
        eta_min=1e-7,
    )
    
    log_file = open("try_log.txt", "w")
    
    def log_model_info(model, log_file):
        log_file.write("=" * 60 + "\n")
        log_file.write("Model Configuration:\n")
        log_file.write(f"max_atoms    = {model.max_atoms}\n")
        log_file.write(f"num_cond     = {model.num_cond}\n")
        log_file.write(f"d_model      = {model.d_model}\n")
        log_file.write(f"nhead        = {model.num_heads}\n")
        log_file.write(f"num_layers   = {model.num_layers}\n")
        log_file.write(f"dim_feedforward = {model.dim_feedforward}\n")
        log_file.write(f"dropout      = {model.dropout}\n")
        log_file.write("\nModel Architecture:\n")
        log_file.write(str(model) + "\n")
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_file.write(f"\nTotal trainable parameters: {total_params}\n")
        log_file.write("=" * 60 + "\n")
        log_file.flush()

    log_model_info(model, log_file)

    best_val = float("inf")
    start_epoch = 0

    if ckpt_path is not None and os.path.exists(ckpt_path):

        ckpt = torch.load(
            ckpt_path,
            map_location=device
        )

        model.load_state_dict(
            ckpt["model_state"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state"]
        )

        scheduler.load_state_dict(
            ckpt["scheduler_state"]
        )

        start_epoch = ckpt["epoch"] + 1

        best_val = ckpt["best_val"]

        print(
            f"Resume from epoch "
            f"{start_epoch}"
        )

    for epoch in range(
        start_epoch,
        satopt["training"]["epoch"]
    ):

        run_epoch(
            model,
            train_loader,
            device,
            epoch,
            phase="train",
            optimizer=optimizer,
            scheduler=scheduler,
            log_file = log_file,
        )

        val_loss = run_epoch(
            model,
            val_loader,
            device,
            epoch,
            phase="val",
            log_file = log_file,
        )

        save_checkpoint(
            os.path.join(
                save_dir,
                "last_prop.pt"
            ),
            epoch,
            model,
            optimizer,
            scheduler,
            best_val,
        )

        if val_loss < best_val:

            best_val = val_loss

            save_checkpoint(
                os.path.join(
                    save_dir,
                    "best_prop.pt"
                ),
                epoch,
                model,
                optimizer,
                scheduler,
                best_val,
            )

            print(
                "Saved best model."
            )

    print("\nTesting...\n")

    run_epoch(
        model,
        test_loader,
        device,
        satopt["training"]["epoch"],
        phase="test",
        log_file = log_file,
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds = CrystalStructureDataset(
        args.train_json,
        randomchose=False,
        dim=2,
        scale=1,
    )

    val_ds = CrystalStructureDataset(
        args.val_json,
        randomchose=False,
        dim=2,
        scale=1,
    )

    test_ds = CrystalStructureDataset(
        args.test_json,
        randomchose=False,
        dim=2,
        scale=1,
    )
    
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

    train_dataset = SequenceStructureDataset(
        train_ds,
        max_atoms=70,
    )

    val_dataset = SequenceStructureDataset(
        val_ds,
        max_atoms=70,
    )

    test_dataset = SequenceStructureDataset(
        test_ds,
        max_atoms=70,
    )

    train_loader = DataLoader(
        SATDataLoader(
            train_dataset,
            cutoff=0.5,
            k_hop=1,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        SATDataLoader(
            val_dataset,
            cutoff=0.5,
            k_hop=1,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        SATDataLoader(
            test_dataset,
            cutoff=0.5,
            k_hop=1,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    satopt = {
        "model": {
            "k_dim": 0,
            "max_length": 80,
        },
        "training": {
            "epoch": args.epochs,
            "optimizer": {
                "lr": 1e-3,
            },
        },
    }

    train_prop_model(
        satopt,
        train_loader,
        val_loader,
        test_loader,
        args.save_dir,
    )