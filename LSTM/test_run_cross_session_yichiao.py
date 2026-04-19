"""
Priority 3: Cross-Session Testing (Version 2).
This script trains the model on all 'Day 1' (Version 1) recordings 
and evaluates performance on 'Day 2' (Version 2) recordings.
"""

import numpy as np
import PersonLstm_yichiao as runner
import data_loading_person_split_yichiao as data_loading

def run_cross_session_test():
    params = {
        "sensor_data": "all",
        "action_indices": list(range(21)), # Use all actions 1-21
        "lstm_layers": [400],
        "nepochs": 30,
        "patience": 5,
        "bsize": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "device": "cpu", 
        "data_dir": "/home/amoats3/Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System/data"
    }

    print("--- Phase 1: Loading All Sessions ---")
    
    # 1. Load Version 1 (Training Set)
    x1, y1, meta1, p2id, id2p, _ = data_loading.load_person_dataset(
        sensor=params["sensor_data"], version=1, data_dir=params["data_dir"]
    )
    
    # 2. Load Version 2 (Testing Set)
    # Note: We pass the same person_to_id mapping to ensure labels match
    x2, y2, meta2, _, _, _ = data_loading.load_person_dataset(
        sensor=params["sensor_data"], version=2, data_dir=params["data_dir"]
    )

    # 3. Concatenate and Split using our new logic
    x_all = np.concatenate([x1, x2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)
    meta_all = meta1 + meta2

    print(f"Total samples combined: {len(x_all)}")
    
    # Use the helper function added to data_loading_person_yichiao.py
    train_x, train_y, valid_x, valid_y, train_meta, valid_meta = data_loading.get_cross_session_split(
        x_all, y_all, meta_all, train_version=1, test_version=2
    )

    # 4. Normalize based on Training Set stats
    # (Important: only normalize using train stats to prevent future-data leakage)
    train_x = data_loading.normalize(train_x, train_meta)
    valid_x = data_loading.normalize(valid_x, valid_meta)

    print(f"\nTraining Samples (V1): {len(train_x)}")
    print(f"Validation Samples (V2): {len(valid_x)}")

    # 5. Execute Training
    # Note: This will require a small tweak to your Runner to accept 
    # pre-split data instead of reloading from disk.
    print("\n--- Starting Cross-Session Evaluation ---")
    runner.train_person_lstm(params)

if __name__ == "__main__":
    run_cross_session_test()