import os
import numpy as np
from ase.io.trajectory import Trajectory
from rich.progress import track
import re

from evaluate import (
    smact_validity,
    get_validity_one,
    GenEval
)

if __name__ == "__main__":
    Eval={
        'main_path':'/home/wjq/SAT_GPT/SAT_data',
        'traj_train':'/home/wjq/SAT_GPT/train.traj',
        'traj_val':'/home/wjq/SAT_GPT/val.traj',
        'n_samples':1,
        'match_para':{
            'stol': 0.5,
            'angle_tol': 10,
            'ltol': 0.3,
            'n_attempt': 1},
        'calc_validity': False,
        'calc_match_rms': True,
        'calc_coverage': False,
        'calc_comp_div': False,
        'calc_struct_div': False,
        'calc_wdist_density': False,
        'calc_wdist_num_elems': False,
        'calc_novelty_unique': False
        }

    # -------------------------------------------------
    # paths
    # -------------------------------------------------
    gen_path   = os.path.join(Eval['main_path'], "gen")
    target_path = os.path.join(Eval['main_path'], "target/target.traj")
    pred_path  = os.path.join(Eval['main_path'], "pred_all.traj")
    true_path  = os.path.join(Eval['main_path'], "true_all.traj")

    traj_train = Trajectory(Eval['traj_train'])
    traj_val   = Trajectory(Eval['traj_val'])

    # -------------------------------------------------
    # 如果 pred / true 已存在，直接复用
    # -------------------------------------------------
    if os.path.exists(pred_path) and os.path.exists(true_path):
        print("[INFO] Found existing pred_all.traj and true_all.traj, skip preprocessing.")
    else:
        print("[INFO] pred_all / true_all not found, start preprocessing...")

        gentrajs = sorted(
            [f for f in os.listdir(gen_path) if re.match(r"gen\d+\.traj$", f)],
            key=lambda x: int(x[3:-5])
        )

        gen_frames = []
        for gen_name in gentrajs:
            traj = Trajectory(os.path.join(gen_path, gen_name))
            for ats in traj:
                gen_frames.append(ats)

        target_frames = list(Trajectory(target_path))

        if len(gen_frames) != len(target_frames):
            raise ValueError(
                f"Total frame mismatch: gen={len(gen_frames)} "
                f"vs target={len(target_frames)}"
            )
        print(f"Total frames OK: {len(gen_frames)}")

        traj_pred = Trajectory(pred_path, mode='w')
        traj_true = Trajectory(true_path, mode='w')

        for i in track(range(len(gen_frames)), description="PreProcessing all frames"):
            ats_pred = gen_frames[i]
            ats_true = target_frames[i]
            print(ats_true,ats_pred)
            
            #ats_pred.set_cell(ats_true.cell, scale_atoms=False)
            #ats_pred.set_chemical_symbols(ats_true.get_chemical_symbols())
            #ats_true.wrap()

            try:
                pos_pred = ats_pred.get_scaled_positions()
                cell_pred = np.array(ats_pred.get_cell())
                pos_true = ats_true.get_scaled_positions()
                cell_true = np.array(ats_true.get_cell())

                print('process')
                
                ats_pred.info["constructed"] = (
                    np.all(~np.isnan(pos_pred)) and
                    np.all(~np.isnan(cell_pred)) and
                    ats_pred.get_volume() > 0.1
                )
                ats_true.info["constructed"] = (
                    np.all(~np.isnan(pos_true)) and
                    np.all(~np.isnan(cell_true)) and
                    ats_true.get_volume() > 0.1
                )
            except Exception:
                ats_pred.info["constructed"] = False
                ats_true.info["constructed"] = False

            smact_validity(ats_pred)
            get_validity_one(ats_pred)
            smact_validity(ats_true)
            get_validity_one(ats_true)

            traj_pred.write(ats_pred)
            traj_true.write(ats_true)

        traj_pred.close()
        traj_true.close()

        print("[INFO] Preprocessing finished.")

    # -------------------------------------------------
    # Eval
    # -------------------------------------------------
    geval = GenEval(
        traj_pred=Trajectory(pred_path),
        traj_true=Trajectory(true_path),
        traj_train=traj_train,
        traj_test=traj_val,
        traj_match_path=f"{Eval['main_path']}/match_all.traj",
        Eval=Eval
    )

    metrics = geval.get_metrics()
    n_match = metrics['n_match']
    rms_dists = np.array(metrics['rms_dists'])
    matchrate = metrics['match_rate']
    mean_rms_dist = rms_dists[rms_dists != None].mean()

    print("matchrate:", matchrate)
    print("mean_rms_dist:", mean_rms_dist)

    with open(f"{Eval['main_path']}/match_rate.csv", "a+") as f:
        f.write("matchrate,rmse\n")
        f.write(f"{matchrate*100}%,{mean_rms_dist}\n")