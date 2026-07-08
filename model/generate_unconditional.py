import os
import torch
from torch.utils.data import DataLoader

from SAT_llm import SATGPT
from model import Transformer
from Z_model import CompositionPredictor, CompositionDecoder

from data import (
    SATDataLoader,
    collate_fn,
    SequenceStructureDataset,
    CrystalStructureDataset,
)

from loss import lattice6_to_matrix
from match import preds_to_traj


# =========================================================
# config
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ckpt_path = "SAT_data/last_ckpt.pt"

save_dir = "./manual_gen"

os.makedirs(save_dir, exist_ok=True)

atom_types = 118

BOS_Z = atom_types + 1
EOS_Z = atom_types + 2


# =========================================================
# model config
# =========================================================

satopt = {
    "model": {
        "k_dim": 1,
        "d_model": 512,
        "n_layer": 12,
        "n_head": 8,
        "d_ff": 1024,
        "d_head": 64,
        "coord_emb_dim": 64,
        "max_length": 80,
        "num_atom_types": 118,
    }
}


# =========================================================
# helper
# =========================================================

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


def build_structure_sequence(
    coords,
    Z,
    EOS_Z
):

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

    lengths_seq = torch.full(
        (B,),
        3 * L + 1,
        device=device,
        dtype=torch.long
    )

    return Z_seq, coord_seq_bos, lengths_seq


# =========================================================
# load models
# =========================================================

prop_model = Transformer(
    num_cond=satopt["model"]["k_dim"],
    max_atoms=satopt["model"]["max_length"]
)

prop_model.to(device)

Z_model = CompositionDecoder(
    num_elements=118,
        #max_atoms=52,
        #num_cond=0,
        d_model=384,
        nhead=12,
        num_layers=10,
        dim_feedforward=1536,
        dropout=0.1,
    num_cond=satopt["model"]["k_dim"],
    max_atoms=70#satopt["model"]["max_length"] // 4
)

Z_model.to(device)

coord_satopt = {
        "model": {
            "k_dim":1,
            "d_model": 256,        # ↓
            "n_layer": 8,         # ↓
            "n_head": 8,           # ↓
            "d_ff": 512,          # ok
            "d_head":64,
            "coord_emb_dim": 64,   # ↓
            "max_length":80,
            "num_atom_types":118,
        }
}

model = SATGPT(
    num_atom_types=coord_satopt["model"]["num_atom_types"],
    k_dim=coord_satopt["model"]["k_dim"],
    d_model=coord_satopt["model"]["d_model"],
    n_layer=coord_satopt["model"]["n_layer"],
    n_head=coord_satopt["model"]["n_head"],
    d_head=coord_satopt["model"]["d_head"],
    d_ff=coord_satopt["model"]["d_ff"],
    num_bins=100,
    max_length=coord_satopt["model"]["max_length"]
)

model.to(device)


# =========================================================
# load checkpoint
# =========================================================

ckpt = torch.load(
    ckpt_path,
    map_location=device
)

model_ckpt = torch.load(
    "Coord_data_small/last_ckpt.pt",
    map_location=device
)
model.load_state_dict(
    model_ckpt["model_state"]
)

prop_model_ckpt = torch.load(
    "prop_data/last_prop.pt",
    map_location=device
)
prop_model.load_state_dict(
    prop_model_ckpt["model_state"]
)

Z_ckpt = torch.load(
    "Z_data_normal/last_Z.pt",
    map_location=device
)

Z_model.load_state_dict(
    Z_ckpt["model_state"]
)

model.eval()
prop_model.eval()
Z_model.eval()

print("Checkpoint loaded.")


# =========================================================
# manually set cond
# =========================================================

# =====================================================
# generation count
# =====================================================

num_gen = 1000
cond_value = 3.0
cond = torch.full(
    (num_gen, 1),
    cond_value,
    device=device,
    dtype=torch.float
)
#print("cond =", cond)

with torch.no_grad():
    Z_gen = Z_model.generate(cond=cond,top_k=5,temperature=1.2)
    Z_gen, lengths = build_Z_sequence(Z_gen, BOS_Z, EOS_Z)
    #print(Z_gen)
    prop_preds = prop_model(Z=Z_gen, cond=cond, lengths=lengths, temperature=0.05)  
    #(logits_x, logits_y, logits_z)
    print("Predicted lattice:")
    print(prop_preds)
    
    B, L = Z_gen.shape
    dummy_coords = torch.zeros(
        (B, L, 3),
        device=device
    )
    Z_seq, _, lengths_seq = build_structure_sequence(
        coords=dummy_coords,
        Z=Z_gen,
        EOS_Z=BOS_Z
    )
    
    print("Parsed Z:")
    print(Z_gen[0])

    coord_preds = model.generate(
        Zs=Z_seq,
        lengths=lengths_seq,
        lattice=prop_preds,
        cond=cond
    )
    pred_seq = coord_preds[:, 1:]
    coord_preds = pred_seq.reshape(
        B,
        L,
        3
    ).float()
    print("Generated coords:")
    print(coord_preds.shape)


    lattice_matrix = lattice6_to_matrix(
        prop_preds
    )
    cart_coords = torch.matmul(
        coord_preds,
        lattice_matrix
    )
    preds_to_traj(
        type_logits=Z_gen,
        coord_preds=cart_coords,
        lengths=lengths,
        traj_path=os.path.join(
            save_dir,
            "manual_gen.traj"
        ),
        cell_params=lattice_matrix,
        ignore_index_list=None
    )

print("Generation finished.")