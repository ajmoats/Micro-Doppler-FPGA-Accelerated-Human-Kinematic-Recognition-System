"""
Person-ID baseline using all sensors together, 5-fold evaluation.

Run from repo root:
python LSTM/test_run_person_all_5fold_yichiao.py
"""

import PersonLstm_yichiao as runner

params = {
    "sensor_data": "all",                # use US25 + US33 + US40
    "version": 1,                        # use *_1 only
    "action_indices": list(range(21)),   # actions 1~21 only
    "lstm_layers": [400],
    "folds": [0, 1, 2, 3, 4],            # full 5-fold
    "nepochs": 5,
    "bsize": 16,
    "dropout": 0.5,
    "lr": 1e-3,
    "print_summary": False,              # keep output cleaner
    "save_results": True,
    "experiment_name": "baseline_all_5fold",
}

runner.train_person_lstm(params)
