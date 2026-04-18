"""
Single-action person identification with all sensors.

Run from repo root:
python LSTM/test_run_person_single_action_all_yichiao.py
"""

import PersonLstm_yichiao as runner

params = {
    "sensor_data": "all",
    "version": 1,
    "action_indices": [9],      # Walk in Place (N)
    "window_len": 200,
    "stride": 100,
    "lstm_layers": [400],
    "folds": [0, 1, 2, 3, 4],
    "nepochs": 5,
    "bsize": 32,
    "max_len": 200,             # fixed window length
    "dropout": 0.5,
    "lr": 1e-3,
    "print_summary": False,
}

runner.train_person_lstm(params)