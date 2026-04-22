"""
First smoke test for person identification.

Run from repo root:
python LSTM/test_run_person_yichiao.py
"""

import PersonLstm_yichiao as runner


params = {
    "sensor_data": "US40",          # first try one sensor
    "version": 1,                   # use *_1 only
    "action_indices": list(range(21)),  # actions 1~21
    "lstm_layers": [400],
    "folds": [0],
    "nepochs": 5,                   # smoke test first
    "bsize": 16,
    "dropout": 0.5,
    "lr": 1e-3,
    "print_summary": True,
    "save_results": True,
    "experiment_name": "smoke_us40_fold0",
}

runner.train_person_lstm(params)


# After smoke test succeeds, try this:
#
# params = {
#     "sensor_data": "US40",
#     "version": 1,
#     "action_indices": list(range(21)),
#     "lstm_layers": [400],
#     "folds": [0, 1, 2, 3, 4],
#     "nepochs": 5,
#     "bsize": 16,
#     "dropout": 0.5,
#     "lr": 1e-3,
#     "print_summary": False,
# }
# runner.train_person_lstm(params)
