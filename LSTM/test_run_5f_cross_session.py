import numpy as np
import torch
import PersonLstm_split_yichiao as runner
import data_loading_person_split_yichiao as data_loading

def run_5_fold_cross_validation():
    params = {
        "sensor_data": "all",
        "action_indices": list(range(21)), 
        "lstm_layers": [400],
        "nepochs": 5,                  
        "folds": [0, 1, 2, 3, 4], # This will now be respected
        "bsize": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "device": "mps" if torch.backends.mps.is_available() else "cpu", 
        "data_dir": "/home/amoats3/Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System/data"
    }

    print("--- Phase 1: Loading All Sessions for 5-Fold CV ---")
    
    # 1. Load V1
    x1, y1, m1, p2id, _, _ = data_loading.load_person_dataset(version=1, data_dir=params["data_dir"])
    
    # 2. Load V2 using same person mapping
    x2, y2, m2, _, _, _ = data_loading.load_person_dataset(version=2, data_dir=params["data_dir"], person_to_id=p2id)

    # 3. Combine all data 
    # We no longer call get_cross_session_split() here
    x_all = np.concatenate([x1, x2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)
    meta_all = m1 + m2

    # 4. Normalize the full set
    x_all = data_loading.normalize(x_all, meta_all)

    print(f"\nTotal Samples combined (V1 + V2): {len(x_all)}")
    print(f"Executing cross-validation for folds: {params['folds']}")

    # 5. Run Trainer
    # We pass the full combined dataset. 
    # Ensure runner.train_person_lstm partitions this data internally.
    print("\n--- Starting 5-Fold Evaluation ---")
    runner.train_person_lstm(params, preloaded_data=(x_all, y_all, meta_all))

if __name__ == "__main__":
    run_5_fold_cross_validation()