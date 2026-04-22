"""
Execution script for regularized person-identification experiments.
"""

import PersonLstm_yichiao as runner

def run_regularized_normalized_experiment():
    # Centralized configuration for the experiment
    params = {
        "sensor_data": "US40",
        "version": 1,
        "action_indices": list(range(21)), 
        "lstm_layers": [400],
        
        # --- Regularization & Training Control ---
        "nepochs": 50,          # High limit; Early Stopping will likely trigger earlier
        "early_stopping_patience": 5,
        "lr": 1e-4,             # Choice: Slower learning for biometric signatures
        "weight_decay": 1e-4,   # Choice: L2 penalty to reduce overfitting
        "dropout": 0.5,         # Standard dropout for hidden layers
        "grad_clip_norm": 1.0,
        
        # --- Data Handling ---
        "bsize": 16,
        "max_len": 500,        # Reduced from 1500 to optimize CPU memory/speed
        "folds": [0, 1, 2, 3, 4],
        "seed": 1337,
        "print_summary": True,
        "mask_val": 0.0,
        "save_results": True,
        "experiment_name": "regularized_us40_5fold",
    }

    print("\n" + "="*50)
    print("STARTING: Regularized Biometric Person-ID Experiment")
    print(f"Sensors: {params['sensor_data']} | Regularization: ACTIVE")
    print("="*50 + "\n")

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
