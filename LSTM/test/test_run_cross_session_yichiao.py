"""
Priority 3: Cross-Session Testing (Version 2).
Trains on 'Day 1' (V1) and evaluates on 'Day 2' (V2) for all people and actions.
"""
import numpy as np
import torch
import PersonLstm_split_yichiao as runner
import LSTM.data_loading_person_split_yichiao as data_loading

def run_cross_session_test():
    params = {
        "sensor_data": "25",
        "action_indices": list(range(21)), # All 21 Actions
        "lstm_layers": [400],
        "nepochs": 5,                   # 5 Epochs
        "bsize": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "device": "mps" if torch.backends.mps.is_available() else "cpu", 
        "data_dir": "/home/amoats3/Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System/data"
    }

    print("--- Phase 1: Loading All Sessions ---")
    
    # 1. Load V1 (Train)
    x1, y1, m1, p2id, _, _ = data_loading.load_person_dataset(version=1, data_dir=params["data_dir"])
    
    # 2. Load V2 (Test) using V1 mapping
    x2, y2, m2, _, _, _ = data_loading.load_person_dataset(version=2, data_dir=params["data_dir"], person_to_id=p2id)

    # 3. Create the Session-based Split
    x_all = np.concatenate([x1, x2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)
    meta_all = m1 + m2

    tx, ty, vx, vy, tm, vm = data_loading.get_cross_session_split(x_all, y_all, meta_all)

    # 4. Normalize
    tx = data_loading.normalize(tx, tm)
    vx = data_loading.normalize(vx, vm)

    print(f"\nTraining Samples (V1): {len(tx)}")
    print(f"Validation Samples (V2): {len(vx)}")

    # 5. Run Trainer
    print("\n--- Starting Cross-Session Evaluation ---")
    runner.train_person_lstm(params, preloaded_data=(tx, ty, vx, vy, tm, vm))

if __name__ == "__main__":
    run_cross_session_test()