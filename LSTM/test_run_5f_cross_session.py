import numpy as np
import torch
import multiprocessing
import PersonLstm_split_yichiao as personlstm
import data_loading_person_split_yichiao as data_loading

def worker_task(i, seed, data_payload, params, return_dict):
    """Execution block for a single process."""
    run_id = f"fold{i}_seed{seed}"
    local_params = params.copy()
    local_params["seed"] = seed
    
    print(f"Launching {run_id}...")
    acc = personlstm.train_person_lstm(local_params, data_payload, run_id)
    return_dict[i] = acc

def run_parallel_sessions():
    params = {
        "nepochs": 5, "bsize": 16, "lr": 1e-4,
        "device": "cpu", 
        "data_dir": "/home/amoats3/Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System/data"
    }

    # Load Data Once
    x1, y1, m1, p2id, _, _ = data_loading.load_person_dataset(version=1, data_dir=params["data_dir"])
    x2, y2, m2, _, _, _ = data_loading.load_person_dataset(version=2, data_dir=params["data_dir"], person_to_id=p2id)
    tx, ty, vx, vy, tm, vm = data_loading.get_cross_session_split(np.concatenate([x1, x2]), np.concatenate([y1, y2]), m1+m2)
    data_payload = (data_loading.normalize(tx, tm), ty, data_loading.normalize(vx, vm), vy, tm, vm)

    seeds = [21, 22, 67, 42, 2586]
    
    # Multiprocessing Setup
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
    for i in range(len(seeds)):
        print(f"Run {i} (Seed {seeds[i]}): {return_dict.get(i):.4f}")

if __name__ == "__main__":
    run_parallel_sessions()