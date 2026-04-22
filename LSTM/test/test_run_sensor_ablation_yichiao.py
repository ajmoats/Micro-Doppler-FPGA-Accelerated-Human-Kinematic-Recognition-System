import numpy as np

import JhummaLstm_yichiao as action_runner
import PersonLstm_yichiao as person_runner
import eval_utils_yichiao as eval_utils


SENSORS = ["US25", "US33", "US40", "all"]


def run_action_ablation():
    rows = []
    for sensor in SENSORS:
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
        results = action_runner.train_lstm(params)
        acc = [item["valid_acc"] for item in results]
        rows.append(
            {
                "sensor": sensor,
                "mean_valid_acc": float(np.mean(acc)),
                "std_valid_acc": float(np.std(acc)),
            }
        )

    eval_utils.save_sensor_comparison(
        rows,
        eval_utils.resolve_results_dir("action_recognition", "sensor_ablation"),
        "Action Recognition Sensor Ablation",
    )
    return rows


def run_person_ablation():
    rows = []
    for sensor in SENSORS:
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
        results = person_runner.train_person_lstm(params)
        acc = [item["valid_acc"] for item in results]
        rows.append(
            {
                "sensor": sensor,
                "mean_valid_acc": float(np.mean(acc)),
                "std_valid_acc": float(np.std(acc)),
            }
        )

    eval_utils.save_sensor_comparison(
        rows,
        eval_utils.resolve_results_dir("person_identification", "sensor_ablation"),
        "Person Identification Sensor Ablation",
    )
    return rows


if __name__ == "__main__":
    action_rows = run_action_ablation()
    person_rows = run_person_ablation()
    eval_utils.save_dual_sensor_comparison(
        action_rows,
        person_rows,
        eval_utils.resolve_results_dir("comparisons", "sensor_ablation"),
        "Sensor Ablation Comparison",
    )
