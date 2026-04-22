"""
PyTorch LSTM baseline for the yichiao test version.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import LSTM.data_loading_yichiao as data_loading
import eval_utils_yichiao as eval_utils


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SequenceDataset(Dataset):
    def __init__(self, x, y, lengths):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.lengths[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, num_layers=1, dropout=0.5, num_classes=16):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]             # shape: (batch, hidden_dim)
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


def _parse_lstm_layers(lstm_layers):
    """
    Keeps things simple:
    - [400] works
    - [400, 400] works
    - [800, 400] is rejected in this first clean version
    """
    if not lstm_layers:
        raise ValueError("lstm_layers cannot be empty")

    if len(set(lstm_layers)) != 1:
        raise ValueError(
            "This yichiao PyTorch version currently supports only equal hidden sizes "
            "across layers, e.g. [400] or [400, 400]."
        )

    hidden_dim = lstm_layers[0]
    num_layers = len(lstm_layers)
    return hidden_dim, num_layers


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=None):
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for x_batch, y_batch, lengths in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(x_batch, lengths)
        loss = criterion(logits, y_batch)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        running_loss += loss.item() * y_batch.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == y_batch).sum().item()
        running_total += y_batch.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for x_batch, y_batch, lengths in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        logits = model(x_batch, lengths)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * y_batch.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == y_batch).sum().item()
        running_total += y_batch.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    for x_batch, y_batch, lengths in loader:
        logits = model(x_batch.to(device), lengths.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y_batch.cpu().numpy().tolist())
        y_pred.extend(preds.tolist())

    return y_true, y_pred


def _action_label_names():
    return [f"action_{action_idx + 1}" for action_idx in data_loading.KEEP_ACTIONS]


def train_lstm(user_params=None):
    """
    Minimal safe baseline.
    Returns a list of fold results.
    """
    params = {
        "lstm_layers": [400],
        "nepochs": 1,
        "folds": [0],
        "seed": 1337,
        "dropout": 0.5,
        "max_len": 404,
        "bsize": 50,
        "sensor_data": "US40",   # try US40 first
        "shuffle": True,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_norm": None,
        "early_stopping_patience": None,
        "mask_val": 0.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "raw_data_path": None,
        "print_summary": True,
        "save_results": False,
        "results_root": None,
        "experiment_name": None,
    }

    if user_params:
        params.update(user_params)

    set_seed(params["seed"])
    device = torch.device(params["device"])
    hidden_dim, num_layers = _parse_lstm_layers(params["lstm_layers"])

    print("\n========== yichiao training config ==========")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("=============================================\n")

    print("Loading data...")
    x_all, y_all, batch_ids = data_loading.load_sensor_data(
        sensor=params["sensor_data"],
        raw_data_path=params["raw_data_path"],
    )
    x_all = data_loading.normalize(x_all)

    label_names = _action_label_names()
    experiment_name = params["experiment_name"] or f"sensor_{params['sensor_data'].lower()}"
    experiment_dir = None
    if params["save_results"]:
        experiment_dir = eval_utils.resolve_results_dir(
            task_name="action_recognition",
            experiment_name=experiment_name,
            results_root=params["results_root"],
        )

    results = []
    aggregate_true = []
    aggregate_pred = []

    for fold in params["folds"]:
        print(f"\n===== FOLD {fold} =====")

        train_x, train_y, valid_x, valid_y = data_loading.split_5_folds(
            x_all, y_all, batch_ids, fold=fold
        )

        if params["print_summary"]:
            data_loading.print_dataset_summary(train_x, train_y, name=f"train fold {fold}")
            data_loading.print_dataset_summary(valid_x, valid_y, name=f"valid fold {fold}")

        train_x_pad, train_y_out, train_lengths = data_loading.pad_for_torch(
            train_x, train_y, max_len=params["max_len"], mask_val=params["mask_val"]
        )
        valid_x_pad, valid_y_out, valid_lengths = data_loading.pad_for_torch(
            valid_x, valid_y, max_len=params["max_len"], mask_val=params["mask_val"]
        )

        train_ds = SequenceDataset(train_x_pad, train_y_out, train_lengths)
        valid_ds = SequenceDataset(valid_x_pad, valid_y_out, valid_lengths)

        train_loader = DataLoader(
            train_ds,
            batch_size=params["bsize"],
            shuffle=params["shuffle"],
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=params["bsize"],
            shuffle=False,
            drop_last=False,
        )

        input_dim = train_x_pad.shape[2]
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=params["dropout"],
            num_classes=16,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        history = []
        best_valid_loss = float("inf")
        best_valid_acc = -1.0
        best_epoch = -1
        patience_counter = 0

        for epoch in range(params["nepochs"]):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                grad_clip_norm=params["grad_clip_norm"],
            )
            valid_loss, valid_acc = evaluate(
                model, valid_loader, criterion, device
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "valid_loss": float(valid_loss),
                    "valid_acc": float(valid_acc),
                }
            )

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch + 1

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch + 1}/{params['nepochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
            )

            if (
                params["early_stopping_patience"] is not None
                and patience_counter >= params["early_stopping_patience"]
            ):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        valid_true, valid_pred = collect_predictions(model, valid_loader, device)
        aggregate_true.extend(valid_true)
        aggregate_pred.extend(valid_pred)

        fold_result = {
            "fold": fold,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "valid_loss": float(valid_loss),
            "valid_acc": float(valid_acc),
            "best_valid_acc": float(best_valid_acc),
            "best_epoch": best_epoch,
            "history": history,
            "valid_true": valid_true,
            "valid_pred": valid_pred,
            "label_names": label_names,
        }

        if experiment_dir is not None:
            fold_dir = experiment_dir / f"fold_{fold}"
            eval_utils.save_history(history, fold_dir)
            report = eval_utils.save_classification_report(
                valid_true,
                valid_pred,
                label_names,
                fold_dir,
                metadata={
                    "task": "action_recognition",
                    "sensor": params["sensor_data"],
                    "fold": fold,
                    "params": params,
                },
            )
            fold_result["report_summary_path"] = report["summary_path"]

        results.append(fold_result)

    if experiment_dir is not None and aggregate_true:
        eval_utils.save_classification_report(
            aggregate_true,
            aggregate_pred,
            label_names,
            experiment_dir / "aggregate",
            metadata={
                "task": "action_recognition",
                "sensor": params["sensor_data"],
                "folds": params["folds"],
                "params": params,
            },
        )

    print("\nDone.")
    print("Fold results:")
    for r in results:
        print(
            {
                "fold": r["fold"],
                "valid_acc": r["valid_acc"],
                "best_valid_acc": r["best_valid_acc"],
                "best_epoch": r["best_epoch"],
            }
        )

    return results


def main():
    train_lstm()


if __name__ == "__main__":
    main()
