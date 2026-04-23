"""
Action Subset Experiments for Person Identification.

Tests which ACTION GROUPS give the model the most information about a person's identity.
Runs cross-session evaluation (train on V1, test on V2) for each action group.

Action index reference (0-based):
    0  = Lunges (N)
    1  = Lunges (NE)
    2  = Lunges (NW)
    3  = L. Leg Steps (N)
    4  = R. Leg Steps (N)
    5  = L. Arm Raises, Fwd (N)
    6  = L. Arm Raises, Side (N)
    7  = R. Arm Raises, Fwd (N)
    8  = R. Arm Raises, Side (N)
    9  = Walk in Place (N)
    10 = Walk Facing Fwd (N-S)
    11 = Walk Facing Side (W-E)
    12 = Walk/Pivot (NE-SW)
    13 = Walk/Pivot (NW-SE)
    14 = Jumping Jacks (N)
    15 = Jump Rope (N)
    16 = Body Squats (N)
    17 = Jump Fwd/Bwd (N-S)
    18 = Jump Fwd/Bwd (NE-SW)
    19 = Jump Fwd/Bwd (NW-SE)
    20 = Punch Fwd (N)
"""

import csv
import json
import numpy as np
from pathlib import Path

try:
    import LSTM.PersonLstm_split_gabrielle as personlstm
    import LSTM.data_loading_person_split_yichiao as data_loading
except ModuleNotFoundError:
    import PersonLstm_split_gabrielle as personlstm
    import data_loading_person_split_yichiao as data_loading



# ACTION GROUP DEFINITIONS
# Edit these to try different groupings

ACTION_EXPERIMENTS = {
    "lunges": {
        "indices": [0, 1, 2],
        "description": "Lunges only (N, NE, NW)",
    },
    "leg_steps": {
        "indices": [3, 4],
        "description": "L. Leg Steps + R. Leg Steps",
    },
    "arm_raises_and_punch": {
        "indices": [5, 6, 7, 8, 20],
        "description": "L/R Arm Raises (Fwd + Side) + Punch Fwd",
    },
    "walking": {
        "indices": [9, 10, 11, 12, 13],
        "description": "Walk in Place + Walk Fwd + Walk Side + Walk/Pivot x2",
    },
    "jumping": {
        "indices": [14, 15, 17, 18, 19],
        "description": "Jumping Jacks + Jump Rope + Jump Fwd/Bwd x3",
    },
    "all_actions": {
        "indices": list(range(21)),
        "description": "All 21 actions (baseline reference)",
    },
}



# BASE TRAINING PARAMS

BASE_PARAMS = {
    "sensor_data": "all",
    "lstm_layers": [400],
    "nepochs": 5,
    "folds": [0],       # single fold for speed; change to list(range(5)) for full eval
    "seed": 1337,
    "dropout": 0.5,
    "bsize": 16,
    "max_len": None,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "mask_val": 0.0,
    "device": "cpu",    # change to "cuda" if GPU available
    "data_dir": None,   # change to "../data" if needed
    "print_summary": False,
    "save_results": True,
    "results_root": "results",
}


def load_cross_session_data(action_indices, sensor, data_dir):
    """
    Load V1 and V2 data for given action indices,
    then split into cross-session train/test sets.
    Normalization is computed from training data only — no leakage.
    """
    print(f"  Loading V1 data for actions {action_indices}...")
    x1, y1, m1, p2id, id2p, _ = data_loading.load_person_dataset(
        sensor=sensor,
        version=1,
        action_indices=action_indices,
        data_dir=data_dir,
    )

    print(f"  Loading V2 data for actions {action_indices}...")
    x2, y2, m2, _, _, _ = data_loading.load_person_dataset(
        sensor=sensor,
        version=2,
        action_indices=action_indices,
        data_dir=data_dir,
        person_to_id=p2id,  # reuse same person mapping so labels match
    )

    # Combine then split by version
    x_all = np.concatenate([x1, x2], axis=0)
    y_all = np.concatenate([y1, y2], axis=0)
    meta_all = m1 + m2

    tx, ty, vx, vy, tm, vm = data_loading.get_cross_session_split(x_all, y_all, meta_all)

    # Normalize using TRAINING stats only
    train_flat = np.concatenate(tx, axis=0)
    mu  = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std == 0] = 1.0
    tx_norm = np.array([(s - mu) / std for s in tx], dtype=object)
    vx_norm = np.array([(s - mu) / std for s in vx], dtype=object)

    print(f"  Train samples: {len(tx_norm)} | Test samples: {len(vx_norm)} | People: {len(p2id)}")

    return (tx_norm, ty, vx_norm, vy, tm, vm), id2p


def run_action_subset_experiments():
    """
    Main entry point. Runs each action group experiment sequentially
    and saves a CSV + JSON summary at the end.
    """
    summary_rows = []

    for exp_name, exp_cfg in ACTION_EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT : {exp_name}")
        print(f"  Actions  : {exp_cfg['description']}")
        print(f"  Indices  : {exp_cfg['indices']}")
        print(f"{'='*60}")

        params = BASE_PARAMS.copy()
        params["experiment_name"] = f"action_subset_{exp_name}"

        # Load and normalize data for this action subset
        try:
            data_payload, id2p = load_cross_session_data(
                action_indices=exp_cfg["indices"],
                sensor=params["sensor_data"],
                data_dir=params["data_dir"],
            )
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            summary_rows.append({
                "experiment":     exp_name,
                "description":    exp_cfg["description"],
                "num_actions":    len(exp_cfg["indices"]),
                "best_valid_acc": None,
                "status":         f"LOAD FAILED: {e}",
            })
            continue

        # Run training
        try:
            results = personlstm.train_person_lstm(
                user_params=params,
                preloaded_data=data_payload,
            )
            best_acc = max(r["best_valid_acc"] for r in results)
            print(f"  >>> {exp_name} best valid acc: {best_acc:.4f}")

            summary_rows.append({
                "experiment":     exp_name,
                "description":    exp_cfg["description"],
                "num_actions":    len(exp_cfg["indices"]),
                "best_valid_acc": float(best_acc),
                "status":         "OK",
            })

        except Exception as e:
            print(f"  ERROR during training: {e}")
            summary_rows.append({
                "experiment":     exp_name,
                "description":    exp_cfg["description"],
                "num_actions":    len(exp_cfg["indices"]),
                "best_valid_acc": None,
                "status":         f"TRAIN FAILED: {e}",
            })

    
    # SAVE RESULTS TO CSV AND JSON
    
    results_dir = Path(BASE_PARAMS["results_root"])
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "action_subset_comparison.csv"
    csv_fields = ["experiment", "description", "num_actions", "best_valid_acc", "status"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "experiment":     row["experiment"],
                "description":    row["description"],
                "num_actions":    row["num_actions"],
                "best_valid_acc": f"{row['best_valid_acc']:.4f}" if row["best_valid_acc"] is not None else "FAILED",
                "status":         row["status"],
            })

    json_path = results_dir / "action_subset_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nCSV  saved to : {csv_path}")
    print(f"JSON saved to : {json_path}")

    return summary_rows


if __name__ == "__main__":
    run_action_subset_experiments()