import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import write
import torch
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


def cryst_seq_ele_to_atoms(cryst_seq_ele):
    """
    从 seq token 格式构建 ASE Atoms（坐标已是 Cartesian）

    Parameters
    ----------
    cryst_seq_ele : list
        如 ['<atom>', 'Ele0', 'F', 'x0', 0.0, ...]

    Returns
    -------
    atoms : ase.Atoms
    """

    symbols = []
    positions = []

    i = 0
    while i < len(cryst_seq_ele):

        token = cryst_seq_ele[i]

        # -------------------------
        # 解析原子块
        # -------------------------
        if token == "<atom>":

            symbol = cryst_seq_ele[i + 2]

            x = float(cryst_seq_ele[i + 4])
            y = float(cryst_seq_ele[i + 6])
            z = float(cryst_seq_ele[i + 8])

            symbols.append(symbol)
            positions.append([x, y, z])

            i += 9
            continue

        # -------------------------
        # 解析晶格参数
        # -------------------------
        if token == "<lattice>":

            cellpar = [
                float(cryst_seq_ele[i + 2]),  # a
                float(cryst_seq_ele[i + 4]),  # b
                float(cryst_seq_ele[i + 6]),  # c
                float(cryst_seq_ele[i + 8]),  # alpha
                float(cryst_seq_ele[i +10]),  # beta
                float(cryst_seq_ele[i +12]),  # gamma
            ]

            break

        i += 1

    # 构建晶胞
    cell = cellpar_to_cell(cellpar)

    atoms = Atoms(
        symbols=symbols,
        positions=np.array(positions),   # ⚠️ Cartesian
        cell=cell,
        pbc=True
    )

    return atoms

import os
def dataset_to_traj(
    dataset,
    traj_filename,
):

    atoms_list = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        cryst_seq_ele = sample['cryst_seq_ele']

        atoms = cryst_seq_ele_to_atoms(
            cryst_seq_ele,
        )
        #print(atoms.get_positions())
        atoms_list.append(atoms)

    out_dir = os.path.dirname(traj_filename)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    write(traj_filename, atoms_list)

def preds_to_traj(
    type_logits,
    coord_preds,
    lengths,
    traj_path,
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

        symbols = [
            id2elem[int(t)]
            for t in type_logits[b, matoms:natoms]
            .detach()
            .cpu()
            .numpy()
        ]

        positions = (
            coord_preds[b, matoms:natoms]
            .detach()
            .cpu()
            .numpy()
        )

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

    out_dir = os.path.dirname(traj_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    write(traj_path, atoms_list)


