import JhummaLstm_yichiao as runner
import eval_utils_yichiao as eval_utils
import numpy as np

sensor_list = ["US25", "US33", "US40", "all"]
rows = []

for sensor in sensor_list:
    print("\n" + "=" * 60)
    print(f"Running sensor setting: {sensor}")
    print("=" * 60)

    params = {
        "sensor_data": sensor,
        "lstm_layers": [400],
        "folds": [0, 1, 2, 3, 4],
        "nepochs": 5,
        "bsize": 50,
        "max_len": 404,
        "dropout": 0.5,
        "lr": 1e-3,
        "print_summary": False,
        "save_results": True,
        "experiment_name": f"sensor_ablation_{sensor.lower()}",
    }

    results = runner.train_lstm(params)
    valid_acc = [item["valid_acc"] for item in results]
    rows.append(
        {
            "sensor": sensor,
            "mean_valid_acc": float(np.mean(valid_acc)),
            "std_valid_acc": float(np.std(valid_acc)),
        }
    )

eval_utils.save_sensor_comparison(
    rows,
    eval_utils.resolve_results_dir("action_recognition", "sensor_ablation"),
    "Action Recognition Sensor Ablation",
)
