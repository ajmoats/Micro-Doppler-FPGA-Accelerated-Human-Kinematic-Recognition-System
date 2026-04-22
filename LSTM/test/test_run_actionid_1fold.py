import JhummaLstm_actionid_yichiao as runner

params = {
    "sensor_data": "all",                # use US25 + US33 + US40
    "nepochs": 5,
    "folds": [0],                        # Single fold for quick check
    "bsize": 50,
    "lr": 1e-3,
    "seed": 1337,        
    "device": "cpu",
    "dropout": 0.5,
    "print_summary": True,
}

print("Starting 1-Fold Action Recognition Test with Actor Breakdown...")
results = runner.train_lstm(params)

# The console will now display:
# --- Fold 0 Participant Breakdown ---
# Actor 00: 98.2% (55/56)
# ...
