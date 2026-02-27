import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import torch
import random
seed = 2025
random.seed(seed)

element_map = {
        "H": 1, "He": 2,
        "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
        "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
        "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
        "As": 33, "Se": 34, "Br": 35, "Kr": 36,
        "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
        "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
        "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
        "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61,
        "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68,
        "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75,
        "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
        "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
        "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93,
        "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
        "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106,
        "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
        "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
    }

def lattice_matrix(a, b, c, alpha, beta, gamma):
                alpha = np.radians(alpha)
                beta  = np.radians(beta)
                gamma = np.radians(gamma)

                A = np.zeros((3,3))
                A[0,0] = a
                A[0,1] = b * np.cos(gamma)
                A[0,2] = c * np.cos(beta)

                A[1,1] = b * np.sin(gamma)
                A[1,2] = c * (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)

                A[2,2] = c * np.sqrt(
                    1
                    - np.cos(beta)**2
                    - ((np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma))**2
                )
                return A
# dataset defined for 
class CrystalStructureDataset(Dataset):
    def __init__(self, json_file, randomchose=True, dim=2, scale=1):
        # 加载 JSON 文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)  # 这里的 self.data 是一个二维列表
        self.random = randomchose
        self.dim = dim
        if scale < 1:
            total = len(self.data)
            keep = int(total * scale)
            # 随机选择保留的样本 index
            indices = random.sample(range(total), keep)
            # 按 index 重新构造数据
            self.data = [self.data[i] for i in indices]
        
    def __len__(self):
        # 数据集的长度为 json 文件中二维列表的行数
        return len(self.data)

    def __getitem__(self, idx):
        # 随机从每一行中选择一个对象
        row = self.data[idx]
        variants = row#["variants"] #此处是值得修改的内容，对于不同的数据包往往在此报错

        if self.random and self.dim == 2:
            sampled_object = random.choice(variants)  # 随机从当前行的多个对象中选择一个
        elif self.random == False and self.dim == 2:
            sampled_object =variants[0]
        else:
            sampled_object = variants
        #return sampled_object
        import numpy as np
        def preprocess_object(obj):
            elements = []
            seq_Ele = []
            crystPrompt = []
            atom_idx = 0
            # ------------- (A) 构造晶胞矩阵 -----------------

            def frac2cart(frac, A):
                return A @ frac

            # 构造晶胞矩阵 A
            A = lattice_matrix(
                obj["a"], obj["b"], obj["c"],
                obj["alpha"], obj["beta"], obj["gamma"]
            )

            lattice_params = [obj['a'], obj['b'], obj['c'], obj['alpha'], obj['beta'], obj['gamma']]

            # ---------------------------------------------------
            # (B) 遍历所有原子，并转换 fractional → Cartesian
            # ---------------------------------------------------
            while f'x{atom_idx}' in obj:
                element = obj[f'Ele{atom_idx}']
                elements.append(element)

                # fractional coords
                fx = float(obj[f'x{atom_idx}'])
                fy = float(obj[f'y{atom_idx}'])
                fz = float(obj[f'z{atom_idx}'])

                frac = np.array([fx, fy, fz], dtype=float)
                cart = frac2cart(frac, A)  # 转换为绝对坐标

                cx, cy, cz = frac.tolist()

                # ---------------------------------------------------
                # (C) 存入 seq_Ele（保持原结构，但用绝对坐标）
                # ---------------------------------------------------
                seq_Ele.extend([
                    f"<atom>",
                    f"Ele{atom_idx}",
                    element,
                    f"x{atom_idx}", cx,
                    f"y{atom_idx}", cy,
                    f"z{atom_idx}", cz
                ])

                atom_idx += 1

            # ---------------------------------------------------
            # (D) species 和晶格参数（保持不变）
            # ---------------------------------------------------
            species, num_atoms = np.unique(elements, return_counts=True)
            species = species.tolist()
            total_atoms = sum(num_atoms.tolist())
            num_atoms = list(map(str, num_atoms))

            # 晶格参数加入序列
            seq_Ele.extend([
                "<lattice>",
                "a", obj["a"], "b", obj["b"], "c", obj["c"],
                "alpha", obj["alpha"], "beta", obj["beta"], "gamma", obj["gamma"],
                "<eos>"
            ])

            # prompt 信息（保持原样）
            crystPrompt.extend(['species'] + species + [
                'total_atoms', str(total_atoms),
                'num_atoms'
            ] + num_atoms + ['<sep>'])

            return {
                'num_atom': num_atoms,
                'species': species,
                'prompt': crystPrompt,
                'cryst_seq_ele': seq_Ele,
                'crystal_json': obj
            }

        # 对采样到的对象进行预处理
        return preprocess_object(sampled_object)

def parse_atoms(data_item):
    """
    从 'cryst_seq_ele' 提取：
    - Z_list: List[int]，原子序数
    - coord_list: List[List[float]]，Cartesian 坐标
    - cellpar: [a, b, c, alpha, beta, gamma]
    """
    seq_Ele = data_item["cryst_seq_ele"]  # 已经是 Cartesian

    Z_list = []
    coord_list = []

    # seq_Ele 结构: ["<atom>", "Ele0", "C", "x0", x, "y0", y, "z0", z, "<atom>", ...]
    i = 0
    while i < len(seq_Ele):
        if seq_Ele[i] == "<atom>":
            ele_symbol = seq_Ele[i+2]
            Z_list.append(element_map[ele_symbol])

            # Cartesian 坐标
            x = float(seq_Ele[i+4])
            y = float(seq_Ele[i+6])
            z = float(seq_Ele[i+8])
            coord_list.append([x, y, z])

            i += 9  # 跳到下一个原子
        else:
            i += 1

    # 晶格参数仍然从 crystal_json 里读（没有变化）
    cj = data_item["crystal_json"]
    cellpar = [
        float(cj["a"]),
        float(cj["b"]),
        float(cj["c"]),
        float(cj["alpha"]),
        float(cj["beta"]),
        float(cj["gamma"]),
    ]
    cond = [
        float(cj.get("formation_energy", 0.0)),
        float(cj.get("energy_above_hull", 0.0)),
        float(cj.get("spacegroup_number", 0.0)),
    ]

    return Z_list, coord_list, cellpar, cond


# -----------------------
# Simple dataset skeleton
# -----------------------
class SequenceStructureDataset(Dataset):
    """
    每个样本：
    - Zs: [L] (atomic numbers or atom type ids)
    - coords: [L, 3] (float coordinates)
    We will produce tokens: [Z1, coord1, Z2, coord2, ...] flattened
    For training autogressive, we provide input seq and target seq shifted by one atomic token pair.
    """
    def __init__(self, examples, max_atoms=64):
        # examples: list of dicts with keys 'Z' (list ints), 'coord' (Nx3 numpy/torch)
        self.examples = examples
        self.max_atoms = max_atoms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        data = self.examples[idx]
        # 直接解析
        #print(data)
        Z, coords, cellpar, cond = parse_atoms(data)
        cellpar = lattice_matrix(cellpar[0],cellpar[1],cellpar[2],cellpar[3],cellpar[4],cellpar[5])

        Z = torch.tensor(Z, dtype=torch.long)          # [L]
        coords = torch.tensor(coords, dtype=torch.float32)  # [L, 3]
        cellpar = torch.tensor(cellpar, dtype=torch.float32).view(-1)
        cond = torch.tensor(cond, dtype=torch.float32)

        L = Z.shape[0]

        # 截断（如果需要）
        if L > self.max_atoms:
            Z = Z[:self.max_atoms]
            coords = coords[:self.max_atoms]
            L = self.max_atoms

        return {
            'Z': Z,
            'coords': coords,
            'L': L,
            'cond': cond,
            'score': 0,
            'num': L,
            'prop': cellpar,
            'par': len(cellpar),
            'condnum': len(cond)
        }

# def collate_batch(batch):
#     # produce batched tokens with padding
#     # We'll implement simple batching: pad to max atoms in batch
#     Bs = len(batch)
#     maxL = max([b['L'] for b in batch])#找出最大原子数
#     Zs = torch.full((Bs, maxL), fill_value=0, dtype=torch.long)  # 后续所有样本的pad
#     coords = torch.zeros((Bs, maxL, 3), dtype=torch.float32)
#     P = torch.zeros((Bs, batch[0]['par']), dtype=torch.float32)
#     Y = []
#     lengths = []
#     num = []
#     pars = []
#     for i, b in enumerate(batch):
#         par = b['par']
#         L = b['L']
#         Zs[i, :L] = b['Z']
#         coords[i, :L] = b['coord']
#         P[i,:par]=b['prop']
#         Y.append(b['score'])
#         lengths.append(L)
#         num.append(b['num'])
#         pars.append(par)
#     return {'Z': Zs, 'coords': coords, 'lengths': torch.tensor(lengths, dtype=torch.long), 'score':Y, 'num':num, 'prop':P, 'par':pars}


class SATDataLoader(Dataset):
    """
    输入数据格式：
    每个样本包含：
        Z: (L,) 原子序列（int）
        coords: (L, 3)
        length: scalar
    """

    def __init__(self, data_list, cutoff=4.0, k_hop=2):
        self.data_list = data_list
        self.cutoff = cutoff
        self.k_hop = k_hop

    def __len__(self):
        return len(self.data_list)

    def extract_khop(self, coords, center):
        """
        使用 PyG 的 k-hop 子图工具
        """
        def k_hop_nodes(center, coords, k, cutoff):
            # coords: (N, 3)
            # 基于欧氏距离截断
            cur_frontier = torch.tensor([center])
            visited = set([center])

            for _ in range(k):
                new_nodes = []
                for c in cur_frontier:
                    d = ((coords - coords[c])**2).sum(dim=1).sqrt()
                    flag  = (d > 0) & (d < cutoff)
                    nbr = torch.where(flag)[0]
                    for x in nbr:
                        if x.item() not in visited:
                            visited.add(x.item())
                            new_nodes.append(x.item())
                cur_frontier = torch.tensor(new_nodes)
            return torch.tensor(sorted(list(visited)))
        
        sub_nodes = k_hop_nodes(
            center,
            coords,
            self.k_hop,
            self.cutoff
        )
        return sub_nodes

    def __getitem__(self, idx):
        data = self.data_list[idx]

        Z = data["Z"].clone().long()        # (L,)
        coords = data["coords"].clone().float()
        P = data["prop"].clone().float()
        par = data['par']
        Y = data["score"]
        cond = data["cond"]
        condnum = data["condnum"]
        length = torch.tensor(len(Z), dtype=torch.long)
        num = data["num"]

        all_sub_nodes = []
        all_indicator = []

        L = Z.size(0) + 1   # +1 给 CLS / condition token

        for center in range(1, L):   # 从 1 开始，跳过 CLS
            sub_nodes = self.extract_khop(coords, center - 1) + 1
            all_sub_nodes.append(sub_nodes)
            all_indicator.append(torch.zeros_like(sub_nodes) + center)


        return {
            "Z": Z,
            "coords": coords,
            "length": length,
            'score': Y,
            'cond': cond,
            'condnum': condnum,
            'num': num,
            'prop': P,
            'par': par,
            "sub_nodes": torch.cat(all_sub_nodes),         # (sum_i |sub_i|)
            "sub_indicator": torch.cat(all_indicator)      # 同样长度
        }

def collate_fn(batch):
    # === Step 1 — padding 主序列（给 Transformer 用） =======================
    Z_list = [item["Z"] for item in batch]
    coords_list = [item["coords"] for item in batch]
    cond_list = [item["cond"] for item in batch]
    P_list = [item["prop"] for item in batch]
    lengths = torch.tensor([len(z) for z in Z_list], dtype=torch.long)
    num = [item["num"] for item in batch]
    score = [item["score"] for item in batch]
    par = [item["par"] for item in batch]
    condnum = [item["condnum"] for item in batch]

    B = len(batch)
    L_max = max(lengths)


    Zs = torch.zeros(B, L_max, dtype=torch.long)
    coords = torch.zeros(B, L_max, 3, dtype=torch.float)
    Prop = torch.zeros(B, par[0], dtype=torch.float)
    Cond = torch.zeros(B, condnum[0], dtype=torch.float)

    for i, (Z, C, P, Co) in enumerate(zip(Z_list, coords_list, P_list, cond_list)):
        L = len(Z)
        Zs[i, :L] = Z
        coords[i, :L] = C
        Prop[i, :par[0]] = P
        Cond[i, :condnum[0]] = Co

    # === Step 2 — subgraph（保持 PyG graph batching 格式） ===================
    sub_nodes_list = []
    sub_indicator_list = []
    sub_batch_list = []

    node_offset = 0
    
    L = 0
    for i, item in enumerate(batch):
        L = max(L, len(item["Z"]))

    for i, item in enumerate(batch):

        sub_nodes = item["sub_nodes"] + node_offset
        sub_indicator = item["sub_indicator"] + node_offset

        sub_nodes_list.append(sub_nodes)
        sub_indicator_list.append(sub_indicator)
        sub_batch_list.append(torch.zeros_like(sub_nodes) + i)

        node_offset += L  # 下一句话的总偏移

    sub_nodes = torch.cat(sub_nodes_list)
    sub_indicator = torch.cat(sub_indicator_list)
    sub_batch_index = torch.cat(sub_batch_list)

    return {
        "Z": Zs,                           # (B, L_max)
        "coords": coords,                  # (B, L_max, 3)
        "lengths": lengths,                # (B,)
        "score": score,
        "cond": Cond,
        "num": num,
        'prop': Prop,
        'par': par,
        'condnum': condnum,
        "sub_nodes": sub_nodes,            # (sum sub_nodes,)
        "sub_indicator": sub_indicator,    # (sum sub_nodes,)
        "sub_batch_index": sub_batch_index # (sum sub_nodes,)
    }

def data_add(train_ds):
    dataset = []
    cellpars = []
    for data in (train_ds):    
        Z, coords, cellpar = parse_atoms(data)
            
        L = len(Z)
        pad =  torch.full((L, 3), -100.0)   # shape (L, 3)，值全为 -100
        coords = torch.tensor(coords, dtype=torch.float32)
        coords = torch.cat([pad, coords], dim=0).numpy()   # 形状 (L+1, 3)
        dataset.append({'Z': Z+Z, 'coord': coords, 'score': len(Z), 'num':L, 'prop':cellpar, 'par':len(cellpar)})
        cellpars.append(cellpar)
    return dataset, cellpars
