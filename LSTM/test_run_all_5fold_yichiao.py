import JhummaLstm_yichiao as runner

params = {
    "sensor_data": "all",
    "lstm_layers": [400],
    "folds": [0, 1, 2, 3, 4],
    "nepochs": 5,
    "bsize": 50,
    "max_len": 404,
    "dropout": 0.5,
    "lr": 1e-3,
    "print_summary": False,
}

runner.train_lstm(params)