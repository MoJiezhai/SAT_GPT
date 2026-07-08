import os
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from data import (
    SATDataLoader,
    collate_fn,
    SequenceStructureDataset,
    CrystalStructureDataset,
)

from Z_model import CompositionDecoder



ATOM_TYPES = 118

PAD_Z = 0
BOS_Z = ATOM_TYPES + 1
EOS_Z = ATOM_TYPES + 2

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val,
):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
        if scheduler is not None else None,
        "best_val": best_val,
    }

    torch.save(ckpt, path)

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
                    #print(Z_repeat,Z_seq)
                Z_seq[:, 1:] = Z_repeat
                B, L, _ = coords.shape
                coord_seq = coords.reshape(B, -1)
                coord_seq_bos = torch.zeros(
                        (B, 3 * L + 1),
                        device=device,
                        dtype=coords.dtype
                    )
                coord_seq_bos[:, 1:] = coord_seq
                lengths_seq = L * 3 + 1
                return Z_seq, coord_seq_bos, lengths_seq
                
def build_composition_sequence(
    Z,
    lengths,
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

    # -----------------------------------
    # BOS
    # -----------------------------------

    Z_seq[:, 0] = BOS_Z

    # -----------------------------------
    # atom tokens
    # -----------------------------------

    Z_seq[:, 1:L+1] = Z

    # -----------------------------------
    # EOS position
    # -----------------------------------

    eos_pos = lengths + 1

    batch_idx = torch.arange(
        B,
        device=device
    )

    Z_seq[
        batch_idx,
        eos_pos
    ] = EOS_Z

    # -----------------------------------
    # PAD after EOS
    # -----------------------------------

    positions = torch.arange(
        seq_len,
        device=device
    ).expand(B, seq_len)

    pad_mask = positions > eos_pos[:, None]

    Z_seq[pad_mask] = PAD_Z

    return Z_seq
                
def build_Z_sequence(
                    Z,
                    BOS_Z,
                    EOS_Z,
                    PAD_Z=0
                ):
                    """
                    Automatically parse BOS/EOS from Z.

                    Input
                    -----
                    Z:
                        [B, T]

                    Possible formats:

                        1.
                        BOS A B C EOS PAD PAD

                        2.
                        A B C EOS PAD PAD

                        3.
                        A B C PAD PAD

                    Returns
                    -------
                    Z_gen:
                        [B, Tmax]

                        cropped atom-only sequence:
                            A B C PAD PAD

                    lengths:
                        [B]

                        valid atom count
                    """

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

                            # ---------------------------------
                            # find BOS
                            # ---------------------------------
                            bos_pos = (
                                valid == BOS_Z
                            ).nonzero(as_tuple=True)[0]

                            if len(bos_pos) > 0:

                                start = bos_pos[0].item() + 1

                            else:

                                start = 0

                            valid = valid[start:]

                            # ---------------------------------
                            # find EOS
                            # ---------------------------------
                            eos_pos = (
                                valid == EOS_Z
                            ).nonzero(as_tuple=True)[0]

                            if len(eos_pos) > 0:

                                end = eos_pos[0].item()

                                parsed = valid[:end]

                            else:

                                parsed = valid

                            # ---------------------------------
                            # remove accidental BOS/EOS inside
                            # ---------------------------------
                            parsed = parsed[
                                (parsed != BOS_Z)
                                & (parsed != EOS_Z)
                            ]

                            atom_len = len(parsed)

                        sequences.append(parsed)

                        lengths.append(atom_len)

                        max_len = max(max_len, atom_len)

                    # -------------------------------------
                    # batch pad
                    # -------------------------------------
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
                

def run_epoch(
    model,
    dataloader,
    device,
    epoch,
    optimizer=None,
    scheduler=None,
    phase="train",
    log_file=None,
):

    assert phase in ["train", "val", "test"]

    is_train = (phase == "train")

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    start_time = timer()

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:

        for step, batch in enumerate(dataloader):

            Z = batch["Z"].to(device)

            cond = batch["cond"].to(device)

            lengths = batch["lengths"].to(device)


            if phase == "train":
                
                Z_use = build_composition_sequence(
                    Z,
                    lengths,
                    BOS_Z,
                    EOS_Z,
                    PAD_Z=PAD_Z
                )
                
                #print(Z_use)
                
                out, loss = model(
                    Z_use,
                    cond=cond,
                    return_loss=True
                )

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                optimizer.step()

                total_loss += loss.item()

            elif phase == "val":

                # ---------------------------------
                # teacher forcing loss
                # ---------------------------------

                out, loss = model(
                    Z,
                    cond=cond,
                    return_loss=True
                )

                total_loss += loss.item()

            elif phase == "test":

                Z_gen = model.generate(
                    cond=cond
                )


    if scheduler is not None and is_train:
        scheduler.step()

    if phase != "test":

        avg_loss = total_loss / len(dataloader)

    else:

        avg_loss = 0.0

    end_time = timer()

    msg = (
        f"[Epoch {epoch}] "
        f"[{phase.upper()}] "
        f"Loss {avg_loss:.6f} | "
        f"Time {end_time - start_time:.2f}s"
    )

    print(msg)

    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()

    return avg_loss


def train_Z_model(
    satopt,
    train_loader,
    val_loader,
    test_loader,
    save_dir,
    ckpt_path=None,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    os.makedirs(save_dir, exist_ok=True)
    log_file = open(
        os.path.join(save_dir, "train_Z_log.txt"),
        "w"
    )
    model = CompositionDecoder(
        num_cond=satopt["model"]["k_dim"],
        max_atoms=satopt["model"]["max_atoms"],
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=satopt["training"]["optimizer"]["lr"],
        weight_decay=satopt["training"]["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=satopt["training"]["epoch"],
        eta_min=1e-7,
    )

    best_val = float("inf")
    start_epoch = 0
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
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
        if (
            scheduler is not None
            and ckpt["scheduler_state"] is not None
        ):
            scheduler.load_state_dict(
                ckpt["scheduler_state"]
            )
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(
            f"Resume epoch {start_epoch}, "
            f"best_val = {best_val:.6f}"
        )

    def log_model_info(model, log_file):
        log_file.write("=" * 60 + "\n")
        log_file.write("Model Configuration:\n")
        log_file.write(f"num_elements = {model.num_elements}\n")
        log_file.write(f"max_atoms    = {model.max_atoms}\n")
        log_file.write(f"num_cond     = {model.num_cond}\n")
        log_file.write(f"d_model      = {model.d_model}\n")
        log_file.write(f"nhead        = {model.nhead}\n")
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
    
    for epoch in range(
        start_epoch,
        satopt["training"]["epoch"]
    ):

        run_epoch(
            model=model,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            phase="train",
            log_file=log_file,
        )

        if epoch%40==1:
            val_loss = run_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            phase="val",
            log_file=log_file,
        )

            if val_loss < best_val:

                best_val = val_loss

                save_checkpoint(
                    path=os.path.join(
                        save_dir,
                        "best_Z.pt"
                    ),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val=best_val,
                )

                print("Saved best_Z.pt")
            
            save_checkpoint(
                    path=os.path.join(
                        save_dir,
                        f"val_Z_epoch_{epoch}.pt"
                    ),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val=best_val,
                )

            print("Saved val_Z.pt")

        save_checkpoint(
            path=os.path.join(
                save_dir,
                "last_Z.pt"
            ),
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val=best_val,
        )

    test_loss = run_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
        epoch=satopt["training"]["epoch"],
        phase="test",
        log_file=log_file,
    )

    print(f"\nTest Loss = {test_loss:.6f}")

    log_file.close()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_json",
        type=str,
        required=True
    )

    parser.add_argument(
        "--val_json",
        type=str,
        required=True
    )

    parser.add_argument(
        "--test_json",
        type=str,
        required=True
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./Z_ckpt"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2025
    )

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
            "k_dim": 1,
            "max_atoms": 70,
        },
        "training": {
            "epoch": args.epochs,
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 1e-4,
            }
        }
    }
    train_Z_model(
        satopt=satopt,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=args.save_dir)#,ckpt_path=os.path.join(args.save_dir,"last_Z.pt"))