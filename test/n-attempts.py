import torch
import os
from ase import Atoms
from ase.io.trajectory import Trajectory
from tqdm import tqdm
from ase.io import write
import torch
import numpy as np

from SAT import SATGPT   # 你的模型定义
from data import SATDataLoader, collate_fn, SequenceStructureDataset, CrystalStructureDataset
from torch.utils.data import DataLoader
from match import cellpar_to_cell, preds_to_traj
from model import Transformer
from loss import lattice6_to_matrix

id2elem = {
    1: "H", 2: "He",
    3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge",
    33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe",
    55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm",
    62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er",
    69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re",
    76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb",
    83: "Bi", 84: "Po", 85: "At", 86: "Rn",
    87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np",
    94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg",
    107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds", 111: "Rg", 112: "Cn",
    113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
}

def preds_to_traj(
     type_logits,
     coord_preds,
     lengths,
     cell_params,
     ignore_index_list=None,
 ):
    device = type_logits.device
    B, N, _ = coord_preds.shape

    if ignore_index_list is None:
        ignore_index_list = [0] * B

    if isinstance(ignore_index_list, torch.Tensor):
        ignore_index_list = ignore_index_list.tolist()

    atoms_list = []

    for b in range(B):

        natoms = int(lengths[b])
        matoms = int(ignore_index_list[b])

        # -------------------------
        # element symbols
        # -------------------------
        symbols = [
            id2elem[int(t)]
            for t in type_logits[b, matoms:natoms]
            .detach()
            .cpu()
            .numpy()
        ]

        # -------------------------
        # positions (Cartesian)
        # -------------------------
        positions = (
            coord_preds[b, matoms:natoms]
            .detach()
            .cpu()
            .numpy()
        )

        # -------------------------
        # NEW: lattice vectors
        # -------------------------
        lat = cell_params[b].detach().cpu().numpy()
        lat = np.asarray(lat, dtype=np.float64).reshape(3, 3)
        
        # 防止训练早期出现负体积
        if np.linalg.det(lat) < 0:
            lat[2] *= -1

        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=lat,
            pbc=True
        )
        atoms_list.append(atoms)

    return atoms_list

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证完全确定性（如果你需要）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# config
# -----------------------
ckpt_path = "best.pt"
out_dir = "./gen"
test_path = "./test.json"
os.makedirs(out_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

N_ATTEMPT = 3        # 1 → 一次生成；20 → 二十次生成

# -----------------------
# load model
# -----------------------
ckpt = torch.load(ckpt_path, map_location=device)

satopt = {
        "model": {
            "num_atom_types":118,
            #"y_dim":6, 
            "k_dim":3,
            "d_model":1024, 
            "n_layer":24, 
            "n_head":16, 
            "d_head":64,
            "d_ff":1024, 
            "coord_emb_dim":512, 
            "max_length":50
        }

        }
model = SATGPT(num_atom_types=satopt["model"]["num_atom_types"], k_dim=satopt["model"]["k_dim"], d_model=satopt["model"]["d_model"], n_layer=satopt["model"]["n_layer"], n_head=satopt["model"]["n_head"], d_head=satopt["model"]["d_head"], d_ff=satopt["model"]["d_ff"], coord_emb_dim=satopt["model"]["coord_emb_dim"], max_length=satopt["model"]["max_length"])
state_dict = torch.load("best.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

prop_model = Transformer(num_cond=satopt["model"]["k_dim"],max_atoms=satopt["model"]["max_length"])
state_dict = torch.load("best_prop.pt", map_location=device)
prop_model.load_state_dict(state_dict)
prop_model.to(device)
prop_model.eval()

# -----------------------
# load test data
# -----------------------
test_ds   = CrystalStructureDataset(test_path,   randomchose=False,dim=2, scale=1)
test_dataset = SequenceStructureDataset(test_ds, max_atoms=70)#Dataset
test_loader = DataLoader(
        SATDataLoader(test_dataset, cutoff=0.5, k_hop=1),
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

# -----------------------
# generation
# -----------------------

BASE_SEED = 2026
for attempt in range(N_ATTEMPT):
    traj_out = Trajectory(
        os.path.join(out_dir, f"gen{attempt}.traj"),
        mode="w"
    )
    
    set_seed(BASE_SEED + attempt)

    for batch in tqdm(test_loader, desc=f"Generate attempt {attempt}"):
        Z = batch['Z'].to(device)
        coords = batch['coords'].to(device)
        lengths = batch['lengths'].to(device)
        prop = batch['prop'].to(device)
        cond = batch['cond'].to(device)
        
        prop_preds = prop_model(Z=Z, cond=cond, lengths=lengths) 
        coord_preds = model.sample(Zs=Z, lattice=prop_preds, cond=cond)
        L = lattice6_to_matrix(prop_preds)

        atoms_list = preds_to_traj(
            type_logits=batch['Z'].to(device),
            coord_preds=coord_preds,
            lengths=batch['lengths'].to(device),
            cell_params=L,
            ignore_index_list=None
        )

        for ats in atoms_list:
            traj_out.write(ats)


    traj_out.close()

print("Generation finished.")
