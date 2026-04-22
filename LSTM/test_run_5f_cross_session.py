import numpy as np
import torch
import multiprocessing
import LSTM.PersonLstm_split_gabrielle as personlstm 
import data_loading_person_split_yichiao as data_loading

def worker_task(i, seed, data_payload, params, return_dict):
    """Execution block for a single process."""
    run_id = f"fold{i}_seed{seed}"
    local_params = params.copy()
    local_params["seed"] = seed
    
    print(f"Launching {run_id}...")
    try:
        # Calls updated 3-argument function
        acc = personlstm.train_person_lstm(local_params, data_payload, run_id)
        return_dict[i] = acc
    except Exception as e:
        print(f"Error in {run_id}: {e}")

def run_parallel_sessions():
    params = {
        "nepochs": 5, "bsize": 16, "lr": 1e-4, "weight_decay": 1e-4,
        "device": "cpu", 
        "data_dir": "../data"
    }

    print("--- Phase 1: Loading Separate Sessions ---")
    x1, y1, m1, p2id, _, _ = data_loading.load_person_dataset(version=1, data_dir=params["data_dir"])
    x2, y2, m2, _, _, _ = data_loading.load_person_dataset(version=2, data_dir=params["data_dir"], person_to_id=p2id)

    x_all = np.concatenate([x1, x2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)
    meta_all = m1 + m2
    tx, ty, vx, vy, tm, vm = data_loading.get_cross_session_split(x_all, y_all, meta_all)

    data_payload = (data_loading.normalize(tx, tm), ty, data_loading.normalize(vx, vm), vy, tm, vm)

    seeds = [21, 22, 67, 42, 2586]
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    for i, seed in enumerate(seeds):
        p = multiprocessing.Process(target=worker_task, args=(i, seed, data_payload, params, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n--- Parallel Summary ---")
    for i, seed in enumerate(seeds):
        val = return_dict.get(i)
        if val is not None:
            print(f"Run {i} (Seed {seed}): {val:.4f}")
        else:
            print(f"Run {i} (Seed {seed}): FAILED")

if __name__ == "__main__":
    run_parallel_sessions()