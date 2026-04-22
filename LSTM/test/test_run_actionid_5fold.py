import JhummaLstm_actionid_yichiao as runner

params = {
    "sensor_data": "all",                # use US25 + US33 + US40
    "nepochs": 5,
    "folds": [0, 1, 2, 3, 4],            # Full 5-fold evaluation
    "bsize": 50,
    "lr": 1e-3,
    "seed": 1337,        
    "device": "cpu",
    "dropout": 0.5,
    "print_summary": False,              # Cleaner output for 5 folds
}

print("Starting 5-Fold Action Recognition Baseline with Actor Breakdown...")
results = runner.train_lstm(params)

# This will generate 5 distinct breakdown tables in your terminal.