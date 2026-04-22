"""
Small entry script for the yichiao PyTorch smoke test.
Run from repo root with:

python LSTM/test_run_yichiao.py
"""

import JhummaLstm_yichiao as runner

# First smoke test:
# - one sensor
# - one fold
# - one epoch
# - no fancy settings
params = {
    "sensor_data": "US40",
    "lstm_layers": [400],
    "folds": [0],
    "nepochs": 5,
    "bsize": 50,
    "max_len": 404,
    "dropout": 0.5,
    "lr": 1e-3,
    "print_summary": True,
    "save_results": True,
    "experiment_name": "smoke_us40_fold0",
}

runner.train_lstm(params)

