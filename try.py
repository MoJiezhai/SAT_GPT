import torch
import torch.nn.functional as F
from SAT import SATGPT
from torch.utils.data import DataLoader
from data import SATDataLoader, collate_fn, SequenceStructureDataset, CrystalStructureDataset
import argparse
import os
from model import Transformer
from loss import autoreg_loss, lattice6_to_matrix

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
    prop_optimizer = torch.optim.AdamW(prop_model.parameters(), lr=satopt["training"]["optamizer"]["lr"], weight_decay=1e-5)
    prop_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    prop_optimizer,
    T_max=satopt["training"]["epoch"],
    eta_min=1e-7
)
    # model
    model = SATGPT(num_atom_types=satopt["model"]["num_atom_types"], k_dim=satopt["model"]["k_dim"], d_model=satopt["model"]["d_model"], n_layer=satopt["model"]["n_layer"], n_head=satopt["model"]["n_head"], d_head=satopt["model"]["d_head"], d_ff=satopt["model"]["d_ff"], coord_emb_dim=satopt["model"]["coord_emb_dim"], max_length=satopt["model"]["max_length"])
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
                coords_frac  = batch['coords'].to(device)
                lengths = batch['lengths'].to(device)
                prop = batch['prop'].to(device)
                cond = batch['cond'].to(device)
                
                if phase == "train":
                    prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths)  
                    coord_preds = model(
                        Zs=Z, 
                        lengths=lengths, 
                        lattice=prop,
                        cond=cond,
                        subgraph_node_index=batch['sub_nodes'].to(device), 
                        subgraph_indicator=batch['sub_indicator'].to(device),
                        sub_batch_index=batch['sub_batch_index'].to(device),
                        teacher_forcing= False,
                        noise_std=0.05*epoch
                    )
                elif phase == "val":
                    prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths)  
                    coord_preds = model(Zs=Z, 
                                        lengths=lengths, 
                        lattice=prop_preds,
                        cond=cond,
                        )
                    
                else:
                    prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths) 
                    coord_preds = model(Zs=Z, 
                                        lengths=lengths, 
                                        lattice=prop_preds,
                        cond=cond,
                        )
                    
                loss_coord, loss_prop, rms_loss, accuracy = autoreg_loss(
                        coord_preds,
                        prop_preds,
                        Z,
                        coords_frac,
                        prop,
                        lengths
                    )    

                loss = prop_scale * loss_prop + coord_scale * loss_coord

                if is_train:
                    optimizer.zero_grad()
                    prop_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    torch.nn.utils.clip_grad_norm_(prop_model.parameters(), 1.0)
                    optimizer.step()
                    prop_optimizer.step()

                with torch.no_grad():
                    total_loss += loss.item()
                    total_coord += loss_coord.item()
                    total_prop += loss_prop
                    total_rms += rms_loss.item()
                    total_acc += accuracy

                # -------- 保存轨迹（通常只在 test）--------
                if save_traj:
                    from match import preds_to_traj
                    traj_path_step = traj_path+f"{step:03d}.traj"
                    L = lattice6_to_matrix(prop_preds)
                    preds_to_traj(
                        type_logits=batch['Z'].to(device),
                        coord_preds=coord_preds,
                        lengths=batch['lengths'].to(device),
                        traj_path=traj_path_step,
                        cell_params=L,
                        ignore_index_list=batch['score']
                    )
                step += 1

        if scheduler is not None and is_train:
            scheduler.step()
            prop_scheduler.step()

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
            f"Rms {avg_rms:.4f} | "
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
            torch.save(prop_model.state_dict(), "best_prop.pt")
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
            log_file=log_file,
            batch_size=batch_size,
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
            traj_path="SAT_data/gen/gen",
            batch_size=batch_size,
            log_file=log_file,
            best_val=best_val,
        )
        

    # generation demo
    model.load_state_dict(torch.load("best.pt"))
    prop_model.load_state_dict(torch.load("best_prop.pt"))

    # -------- test --------  
    run_epoch(
        model,
        test_loader,
        device,
        epoch,
        phase="test",
        coord_scale=coord_scale,
        prop_scale=prop_scale,
        batch_size=batch_size,
        save_traj=True,
        traj_path="SAT_data/gen/gen",
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
    train_ds = CrystalStructureDataset(train_path, randomchose=True,dim=2, scale=1)
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
    traj_filename="SAT_data/target/train.traj"
    )
    dataset_to_traj(
    val_ds,
    traj_filename="SAT_data/target/val.traj"
    )
    dataset_to_traj(
    test_ds,
    traj_filename="SAT_data/target/target.traj"
    )
    # print('生成完成')
    
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SATDataLoader(test_dataset, cutoff=0.5, k_hop=1),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    satopt = {
        "model": {
            "num_atom_types":atom_types,
            #"y_dim":6, 
            "k_dim":3,
            "d_model":1024, 
            "n_layer":24, 
            "n_head":16, 
            "d_head":64,
            "d_ff":1024, 
            "coord_emb_dim":512, 
            "max_length":50
        },
        "training": {
            "epoch": args.epochs,
            "optamizer": {
                "lr": 1e-4,
                "weight_decay": 1e-4
            }
        }
    }
    
    toy_example(satopt, train_loader, val_loader, test_loader)
