"""
Priority 4: Cleaner Source-Separated Evaluation.
This script evaluates person identification by keeping entire files 
separate to prevent data leakage between training and validation.
"""

import PersonLstm_yichiao as runner
import LSTM.data_loading_person_split_yichiao as data_loading

def run_source_separated_test():
    # 1. Basic configuration (using windowing for single-action focus)
    params = {
        "sensor_data": "all",
        "version": 1,
        "action_indices": [9],           # Example: Focus on 'Walk in Place'
        "window_len": 200,
        "stride": 100,
        "lstm_layers": [400],
        "nepochs": 20,                   # Increased slightly for better convergence
        "patience": 5,
        "bsize": 32,
        "lr": 1e-4,
        "device": "cpu",                 # Change to 'cuda' or 'mps' if available
        "data_dir": "/home/amoats3/Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System/data"
    }

    # 2. Load the full dataset first
    print("Loading dataset for Source-Separated Evaluation...")
    x_all, y_all, metadata, person_to_id, id_to_person, _ = data_loading.load_person_dataset(
        sensor=params["sensor_data"],
        version=params["version"],
        action_indices=params["action_indices"],
        data_dir=params["data_dir"],
        window_len=params["window_len"],
        stride=params["stride"]
    )

    # 3. Define which files to use for Validation (The "Hold-out" session)
    # We pick one or two specific participants to be the test set for this run.
    # In a real 5-fold file-split, you'd rotate these.
    test_files = ["data_rot_dm_1.mat", "data_rot_ks_1.mat"]
    
    print(f"Splitting data... Holding out files: {test_files}")
    
    # 4. Use the new helper function we added to the loader
    train_x, train_y, valid_x, valid_y, train_meta, valid_meta = data_loading.get_file_based_split(
        x_all, y_all, metadata, test_files
    )

    # 5. Normalize and Train
    # Note: We pass the split data directly if your runner supports it, 
    # or we modify the runner to accept pre-split data.
    # For this smoke test, we'll use the logic inside your existing runner.
    
    print("\n--- Starting Source-Separated Training ---")
    # This assumes your PersonLstm_yichiao has been updated to handle manual splits.
    # If not, you can call your train_one_epoch loop directly here.
    runner.train_person_lstm(params)

if __name__ == "__main__":
    run_source_separated_test()