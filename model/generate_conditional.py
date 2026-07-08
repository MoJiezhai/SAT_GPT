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
from Z_model import CompositionDecoder
from loss import autoreg_loss, lattice6_to_matrix



# -----------------------
# Example usage (toy)
# -----------------------
atom_types = 118#可用的原子种类
from timeit import default_timer as timer
def toy_example(satopt, train_loader, val_loader, test_loader, ckpt_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_file = open("try_log.txt", "w")
    
    prop_model = Transformer(num_cond=satopt["model"]["k_dim"],max_atoms=satopt["model"]["max_length"]//2)
    prop_model.to(device)
    prop_optimizer = torch.optim.AdamW(prop_model.parameters(), lr=satopt["training"]["optimizer"]["lr"], weight_decay=1e-5)
    prop_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    prop_optimizer,
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
        scheduler=None,
        save_traj=False,
        traj_path=None,
        batch_size=None,
        log_file=None,
    ):
        start_time = timer()
        assert phase in ["train", "val", "test"]

        is_train = (phase == "train")

        if is_train:
            model.train()
        else:
            model.eval()

        step = 0

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for batch in dataloader:
                Z = batch['Z'].to(device)
                EOS_Z = atom_types
                coords_frac  = batch['coords'].to(device)
                from data import apply_random_shift
                coords_frac_add, _ = apply_random_shift(coords_frac, 0.3)
                coords_frac = coords_frac_add
                lengths = batch['lengths'].to(device)
                prop = batch['prop'].to(device)
                cond = batch['cond'].to(device)
                B, L, _ = coords_frac.shape
                #print(cond)
                
                def build_structure_sequence(coords, Z, lengths, EOS_Z):
                    B, L, _ = coords.shape
                    device = coords.device
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
                    coord_seq = coords.reshape(B, -1)
                    coord_seq_bos = torch.zeros(
                        (B, 3 * L + 1),
                        device=device,
                        dtype=coords.dtype
                    )
                    coord_seq_bos[:, 1:] = coord_seq
                    lengths_seq = lengths * 3 + 1
                    return Z_seq, coord_seq_bos, lengths_seq
                
                
                Z_seq, coord_seq_bos, lengths_seq = build_structure_sequence(coords=coords_frac,Z=Z,lengths=lengths,EOS_Z=EOS_Z)
                prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths, temperature=0.05)  
                    #(logits_x, logits_y, logits_z)
                coord_preds = model.generate(
                    Zs=Z_seq, 
                    lengths=lengths_seq, 
                    lattice=prop_preds,
                    cond=cond)
                pred_seq = coord_preds[:, 1:]
                coord_preds = pred_seq.reshape(B, L, 3).float() 
                

                # -------- 保存轨迹（通常只在 test）--------
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

       

    # train few steps

    model_ckpt = torch.load(
        "SAT_data_mp52/last_ckpt.pt",
        map_location=device
    )
    model.load_state_dict(
        model_ckpt["model_state"]
    )

    prop_model_ckpt = torch.load(
        "prop_data_mp52/last_prop.pt",
        map_location=device
    )
    prop_model.load_state_dict(
        prop_model_ckpt["model_state"]
    )


    # -------- test --------  
    run_epoch(
        model,
        test_loader,
        device,
        epoch=satopt["training"]["epoch"],
        phase="test",
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
            "k_dim":0,
            "d_model": 512,        # ↓
            "n_layer": 12,         # ↓
            "n_head": 8,           # ↓
            "d_ff": 1024,          # ok
            "d_head":64,
            "coord_emb_dim": 64,   # ↓
            "max_length":160,
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
    )#ckpt_path=os.path.join(args.save_dir, "last_ckpt.pt"))
