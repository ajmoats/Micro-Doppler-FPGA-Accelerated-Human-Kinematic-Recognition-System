"""
Execution script for Normalized Person Identification experiments.
This script bridges the 30% validation gap by enforcing:
1. Source-separated 5-fold cross-validation.
2. Weight Decay and Reduced Learning Rate.
3. Early Stopping based on Validation Loss.
"""

import PersonLstm_norm_yichiao as runner

def run_regularized_normalized_experiment():
    # Centralized configuration for the experiment
    params = {
        "sensor_data": "US40",
        "version": 1,
        "action_indices": list(range(21)), 
        "lstm_layers": [400],
        
        # --- Regularization & Training Control ---
        "nepochs": 50,          # High limit; Early Stopping will likely trigger earlier
        "patience": 5,          # Stop if no val_loss improvement for 5 epochs
        "lr": 1e-4,             # Choice: Slower learning for biometric signatures
        "weight_decay": 1e-4,   # Choice: L2 penalty to reduce overfitting
        "dropout": 0.5,         # Standard dropout for hidden layers
        
        # --- Data Handling ---
        "bsize": 16,
        "max_len": 500,        # Reduced from 1500 to optimize CPU memory/speed
        "folds": [0, 1, 2, 3, 4], # Conduct full 5-fold validation
        "seed": 1337,
        "data_dir": "/home/amoats3/Micro-Doppler-FPGA-Accelerated-Human-Kinematic-Recognition-System/data",
        
        # --- Logging ---
        "print_summary": True,
        "mask_val": 0.0,
    }

    print("\n" + "="*50)
    print("STARTING: Normalized Biometric Person-ID Experiment")
    print(f"Sensors: {params['sensor_data']} | Regularization: ACTIVE")
    print("="*50 + "\n")

    # This calls the train_person_lstm function in your updated LSTM file
    results = runner.train_person_lstm(params)

    # Summary of all folds
    print("\n" + "="*50)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*50)
    
    total_best_acc = 0
    for r in results:
        print(f"Fold {r['fold']}: Best Acc = {r['best_valid_acc']:.4f} (at Epoch {r['best_epoch']})")
        total_best_acc += r['best_valid_acc']
    
    avg_acc = total_best_acc / len(results)
    print(f"\nAverage 5-Fold Cross-Validation Accuracy: {avg_acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    run_regularized_normalized_experiment()