import numpy as np

import PersonLstm_yichiao as runner
import eval_utils_yichiao as eval_utils


sensor_list = ["US25", "US33", "US40", "all"]
rows = []

for sensor in sensor_list:
    print("\n" + "=" * 60)
    print(f"Running person-ID sensor setting: {sensor}")
    print("=" * 60)

    params = {
        "sensor_data": sensor,
        "version": 1,
        "action_indices": list(range(21)),
        "lstm_layers": [400],
        "folds": [0, 1, 2, 3, 4],
        "nepochs": 5,
        "bsize": 16,
        "dropout": 0.5,
        "lr": 1e-3,
        "print_summary": False,
        "save_results": True,
        "experiment_name": f"sensor_ablation_{sensor.lower()}",
    }

    results = runner.train_person_lstm(params)
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
    eval_utils.resolve_results_dir("person_identification", "sensor_ablation"),
    "Person Identification Sensor Ablation",
)
