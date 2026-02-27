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

from SATmain.models import GraphTransformer
import torch_geometric.utils as utils
from prop import build_corrected_lattice
from model import Transformer

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

from loss import autoreg_loss, frac_to_cart,  prop_preds_to_lattice_matrix, lattice6_to_matrix
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
    
    prop_model = Transformer(num_cond=satopt["model"]["k_dim"],max_atoms=satopt["model"]["max_length"])
    prop_model.to(device)
    #print(torch.isnan(prop_model.element_embed.weight).any())
    prop_optimizer = torch.optim.AdamW(prop_model.parameters(), lr=satopt["training"]["optamizer"]["lr"], weight_decay=1e-5)
    prop_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(prop_optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=15,
                                                            min_lr=1e-05,
                                                            verbose=False)
    
    # model
    model = SATGPT(num_atom_types=satopt["model"]["num_atom_types"], k_dim=satopt["model"]["k_dim"], d_model=satopt["model"]["d_model"], n_layer=satopt["model"]["n_layer"], n_head=satopt["model"]["n_head"], d_head=satopt["model"]["d_head"], d_ff=satopt["model"]["d_ff"], coord_emb_dim=satopt["model"]["coord_emb_dim"], max_length=satopt["model"]["max_length"])
    model.to(device)
    
    '''prop_deg = torch.cat([
    utils.degree(data.edge_index[1], num_nodes=data.num_nodes)
    for data in train_dset])
     prop_model = GraphTransformer(in_size=satopt["model"]["num_atom_types"],
    #                          num_class=9,
    #                          d_model=64,
    #                          dim_feedforward=2*64,
    #                          dropout=0.2,
    #                          num_heads=8,
    #                          num_layers=6,
    #                          batch_norm=False,
    #                          abs_pe='rw',
    #                          abs_pe_dim=20,
    #                          gnn_type='pna2',
    #                          use_edge_attr='store_true',
    #                          num_edge_features=10000,
    #                          edge_dim=32,
    #                          k_hop=2,
    #                          se="gnn",
    #                          deg=prop_deg,
    #                          global_pool='mean'
    #                          ) 
    # prop_model.to(device)
    # prop_criterion = nn.L1Loss()
    # prop_optimizer = torch.optim.AdamW(model.parameters(), lr=1, weight_decay=1e-5)
    # prop_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(prop_optimizer, mode='min',
    #                                                         factor=0.5,
    #                                                         patience=15,
    #                                                         min_lr=1e-05,
    #                                                         verbose=False)
    # prop_best_val_loss = float('inf')'''
    
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
        # prop_model = prop_model,
        # prop_criterion = prop_criterion,
        # prop_optimizer = prop_optimizer,
        # prop_lr_scheduler = prop_lr_scheduler,
        # prop_best_val_loss = prop_best_val_loss,
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
                #print(lengths)
                
                # cond = coords_frac.view(coords_frac.size(0), -1)
                # _, current_dim = cond.shape  # 计算非batch维度的总元素数
                # if current_dim < satopt["model"]["k_dim"]:
                #     # 计算需要补齐的0的个数
                #     pad_size = satopt["model"]["k_dim"] - current_dim
                #     # 先展平非batch维度
                #     #print(new)
                #     # 在后面补0 (pad的格式: (左, 右) 对最后一维)
                #     cond = F.pad(cond, (0, pad_size), mode='constant', value=-1)
                #print(cond)
                

                #cond[:, 3:] = (cond[:, 3:] - 60) / 6
                #prop[:, 3:] = (prop[:, 3:] - 60) / 6

                if phase == "train":   
                    #print("element_embed weight:", torch.isnan(prop_model.element_embed.weight).any())
                    #print("global_token:", torch.isnan(prop_model.global_token).any())
                    prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths)  
                    #print(Z)               
                    coord_preds = model(
                        Zs=Z, #coords=coords_frac, 
                        lengths=lengths, 
                        lattice=prop,
                        cond=cond,
                        subgraph_node_index=batch['sub_nodes'].to(device), 
                        subgraph_indicator=batch['sub_indicator'].to(device),
                        sub_batch_index=batch['sub_batch_index'].to(device),
                        #sat=False,
                        teacher_forcing= False,
                        noise_std=0.05*epoch
                    )
                    

                elif phase == "val":
                    prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths)  
                    coord_preds = model(Zs=Z, 
                                        #coords=coords_frac, 
                                        lengths=lengths, 
                        lattice=prop_preds,
                        cond=cond,
                        #sat=False,
                        )#model.sample(Z, cond)  
                    
                else:
                    prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths) 
                    coord_preds = model.sample(Zs=Z, lattice=prop_preds, 
                                               cond=cond)
                
                #if random.random()<0.01:print(coords_frac, coord_preds)  
                
                #print(coords_frac, coord_preds)
                # 
                # coord_preds = torch.linalg.solve(
                #     L,
                #     coord_preds.transpose(1,2)
                # ).transpose(1,2)

                #prop_preds = prop
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
                    #c = timer()
                    optimizer.zero_grad()
                    prop_optimizer.zero_grad()
                    loss.backward()
                    
                    # total_norm = 0.0
                    # count = 0
                    # for p in model.parameters():
                    #     if p.grad is not None:
                    #         total_norm += p.grad.norm().item()
                    #         count += 1

                    # print("avg grad norm:", total_norm / max(count, 1))
                    
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
                    #print(prop_preds, L)
                    #print(L)
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
            #torch.save(prop_model.state_dict(), "best_prop.pt")
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
            prop_scale=prop_scale,#*epoch/satopt["training"]["epoch"],#*epoch/satopt["training"]["epoch"],
            optimizer=optimizer,
            scheduler=scheduler,
            log_file=log_file,
            batch_size=batch_size,
            # prop_model = prop_model,
            # prop_criterion = prop_criterion,
            # prop_optimizer = prop_optimizer,
            # prop_lr_scheduler = prop_lr_scheduler
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
            # prop_model = prop_model,
            # prop_best_val_loss = prop_best_val_loss,
            # prop_criterion = prop_criterion,
        )
        

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
        batch_size=batch_size,
        save_traj=True,
        traj_path="SAT_data/gen/gen",
        log_file=log_file,
        # prop_model = prop_model,
        # prop_criterion = prop_criterion,
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
        SATDataLoader(train_dataset, cutoff=0.1, k_hop=1),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        SATDataLoader(val_dataset, cutoff=0.1, k_hop=1),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SATDataLoader(test_dataset, cutoff=0.1, k_hop=1),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    from prop import convert_train_dataset
    train_dset = convert_train_dataset(train_dataset)
    
    satopt = {
        "model": {
            "num_atom_types":atom_types,
            #"y_dim":6, 
            "k_dim":3,
            "d_model":1024, 
            "n_layer":16, 
            "n_head":12, 
            "d_head":64,
            "d_ff":1024, 
            "coord_emb_dim":512, 
            "max_length":50
        },
        "training": {
            "epoch": args.epochs,
            "optamizer": {
                "lr": 1e-5,
                "weight_decay": 0.01
            }
        }
    }
    
    toy_example(satopt, train_loader, val_loader, test_loader)
