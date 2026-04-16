import JhummaLstm_yichiao as runner

sensor_list = ["US25", "US33", "US40", "all"]

for sensor in sensor_list:
    print("\n" + "=" * 60)
    print(f"Running sensor setting: {sensor}")
    print("=" * 60)

    params = {
        "sensor_data": sensor,
        "lstm_layers": [400],
        "folds": [0],
        "nepochs": 5,
        "bsize": 50,
        "max_len": 404,
        "dropout": 0.5,
        "lr": 1e-3,
        "print_summary": False,
    }

    runner.train_lstm(params)