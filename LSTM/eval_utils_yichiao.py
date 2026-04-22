from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_results_dir(task_name, experiment_name=None, results_root=None):
    if results_root is None:
        results_root = Path(__file__).resolve().parent.parent / "results"
    root = ensure_dir(results_root)
    if experiment_name is None:
        experiment_name = "default"
    return ensure_dir(root / task_name / experiment_name)


def confusion_matrix_from_predictions(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def normalized_confusion_matrix(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )


def per_class_accuracy(matrix):
    row_sums = matrix.sum(axis=1)
    correct = np.diag(matrix)
    return np.divide(
        correct,
        row_sums,
        out=np.zeros_like(correct, dtype=np.float64),
        where=row_sums != 0,
    )


def save_matrix_csv(matrix, labels, out_path):
    out_path = Path(out_path)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + list(labels))
        for label, row in zip(labels, matrix):
            writer.writerow([label] + list(row))


def plot_confusion_matrix(matrix, labels, out_path, title, normalized=False):
    fig_w = max(8, len(labels) * 0.5)
    fig_h = max(6, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    value_fmt = ".2f" if normalized else "d"
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                format(value, value_fmt),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_history(history, out_dir):
    out_dir = ensure_dir(out_dir)
    history_path = out_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if not history:
        return

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    valid_loss = [item["valid_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    valid_acc = [item["valid_acc"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, valid_loss, label="valid")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, valid_acc, label="valid")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(
    y_true,
    y_pred,
    labels,
    out_dir,
    metadata=None,
):
    out_dir = ensure_dir(out_dir)
    num_classes = len(labels)
    matrix = confusion_matrix_from_predictions(y_true, y_pred, num_classes)
    matrix_norm = normalized_confusion_matrix(matrix)
    class_acc = per_class_accuracy(matrix)
    overall_acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    save_matrix_csv(matrix, labels, out_dir / "confusion_matrix.csv")
    save_matrix_csv(np.round(matrix_norm, 6), labels, out_dir / "confusion_matrix_normalized.csv")
    plot_confusion_matrix(matrix, labels, out_dir / "confusion_matrix.png", "Confusion Matrix")
    plot_confusion_matrix(
        matrix_norm,
        labels,
        out_dir / "confusion_matrix_normalized.png",
        "Normalized Confusion Matrix",
        normalized=True,
    )

    summary = {
        "overall_accuracy": overall_acc,
        "num_samples": int(len(y_true)),
        "labels": list(labels),
        "per_class_accuracy": {
            label: float(acc) for label, acc in zip(labels, class_acc)
        },
        "metadata": metadata or {},
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return {
        "overall_accuracy": overall_acc,
        "confusion_matrix": matrix,
        "confusion_matrix_normalized": matrix_norm,
        "per_class_accuracy": class_acc,
        "summary_path": str(out_dir / "summary.json"),
    }


def save_sensor_comparison(rows, out_dir, title):
    out_dir = ensure_dir(out_dir)
    csv_path = out_dir / "sensor_comparison.csv"
    json_path = out_dir / "sensor_comparison.json"
    png_path = out_dir / "sensor_comparison.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sensor", "mean_valid_acc", "std_valid_acc"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    sensors = [row["sensor"] for row in rows]
    means = [row["mean_valid_acc"] for row in rows]
    stds = [row["std_valid_acc"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(sensors, means, yerr=stds, capsize=4, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "png_path": str(png_path),
    }


def save_dual_sensor_comparison(action_rows, person_rows, out_dir, title):
    out_dir = ensure_dir(out_dir)
    png_path = out_dir / "sensor_comparison_tasks.png"

    sensors = [row["sensor"] for row in action_rows]
    action_means = [row["mean_valid_acc"] for row in action_rows]
    person_means = [row["mean_valid_acc"] for row in person_rows]

    x = np.arange(len(sensors))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, action_means, width=width, label="action")
    ax.bar(x + width / 2, person_means, width=width, label="person")
    ax.set_xticks(x)
    ax.set_xticklabels(sensors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"png_path": str(png_path)}
