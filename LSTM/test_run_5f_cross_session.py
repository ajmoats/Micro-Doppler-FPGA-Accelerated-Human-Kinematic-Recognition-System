import numpy as np
import torch
import random
import PersonLstm_split_yichiao as runner
import data_loading_person_split_yichiao as data_loading

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For Mac/MPS users
    if torch.backends.mps.is_available():
        torch.manual_seed(seed)

def run_5_fold_cross_session():
    params = {
        "sensor_data": "all",
        "action_indices": list(range(21)), 
        "lstm_layers": [400],
        "nepochs": 5,                  
        "folds": [0], 
        "bsize": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "device": "mps" if torch.backends.mps.is_available() else "cpu", 
        "data_dir": "/home/amoats3/Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System/data"
    }

    print("--- Phase 1: Loading Separate Sessions ---")
    
    # 1. Load V1 and V2
    x1, y1, m1, p2id, _, _ = data_loading.load_person_dataset(version=1, data_dir=params["data_dir"])
    x2, y2, m2, _, _, _ = data_loading.load_person_dataset(version=2, data_dir=params["data_dir"], person_to_id=p2id)

    # 2. Strict Day 1 -> Day 2 Split
    x_all = np.concatenate([x1, x2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)
    meta_all = m1 + m2
    tx, ty, vx, vy, tm, vm = data_loading.get_cross_session_split(x_all, y_all, meta_all)

    # 3. Normalize
    tx = data_loading.normalize(tx, tm)
    vx = data_loading.normalize(vx, vm)

    # 4. Perform 5 Independent Runs
    seeds = [42, 123, 999, 555, 777] # Explicit seeds for 5 different initializations
    
    for i, seed in enumerate(seeds):
        print(f"\n--- Starting Run {i+1}/5 (Seed: {seed}) ---")
        set_seed(seed)
        
        preloaded_data = (tx, ty, vx, vy, tm, vm)
        runner.train_person_lstm(params, preloaded_data=preloaded_data)

if __name__ == "__main__":
    run_5_fold_cross_session()