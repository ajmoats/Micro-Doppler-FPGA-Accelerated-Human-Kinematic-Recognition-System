"""
PyTorch LSTM for person identification.
Updated to support manual data splits for Source-Separation and Cross-Session testing.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import LSTM.data_loading_person_split_yichiao as data_loading
    import LSTM.eval_utils_yichiao as eval_utils
except ModuleNotFoundError:
    print("Module not found, trying local imports")
    import data_loading_person_split_yichiao as data_loading
    import eval_utils_yichiao as eval_utils

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed_all(seed)


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
    def __init__(self, input_dim, hidden_dim=400, num_layers=1, dropout=0.5, num_classes=10):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        # change dropout depending on models performance
        #    High train acc, low valid acc  → Overfitting  → INCREASE dropout
        #    Low train acc,  low valid acc  → Underfitting → DECREASE dropout
        #    High train acc, high valid acc → Good fit     → Keep it
        self.dropout = nn.Dropout(dropout) # regularization step
        self.fc = nn.Linear(hidden_dim, num_classes) #final classifier

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


def _parse_lstm_layers(lstm_layers):
    if not lstm_layers:
        raise ValueError("lstm_layers cannot be empty")

    if len(set(lstm_layers)) != 1:
        raise ValueError(
            "This version only supports equal hidden sizes across layers, "
            "e.g. [400] or [200, 200]."
        )

    hidden_dim = lstm_layers[0]
    num_layers = len(lstm_layers)
    return hidden_dim, num_layers


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=None):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

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

        total_loss += loss.item() * y_batch.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_count += y_batch.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x_batch, y_batch, lengths in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        logits = model(x_batch, lengths)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * y_batch.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_count += y_batch.size(0)

    return total_loss / total_count, total_correct / total_count


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


def train_person_lstm(user_params=None, manual_split=None):
    """
    Main training entry point.
    If manual_split is provided, the function skips automatic loading/splitting.
    """
    params = {
        "sensor_data": "US40",
        "version": 1,
        "action_indices": list(range(21)),   # actions 1~21 only
        "lstm_layers": [400],
        "nepochs": 1,
        "folds": [0],
        "seed": 1337,
        "dropout": 0.5,
        "bsize": 16,
        "max_len": None,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_norm": None,
        "early_stopping_patience": None,
        "mask_val": 0.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": None,
        "print_summary": True,
        "window_len": None,
        "stride": None,
        "save_results": False,
        "results_root": None,
        "experiment_name": None,
    }

    if user_params:
        params.update(user_params)

    set_seed(params["seed"])
    device = torch.device(params["device"])
    hidden_dim, num_layers = _parse_lstm_layers(params["lstm_layers"])

    print("\n========== person-id training config ==========")
    for k, v in params.items():
        print(f"{k}: {v}")
    print("===============================================\n")

    # DATA HANDLING
    if manual_split is not None:
        print(">>> Using manual data split (Source-Separated or Cross-Session)")
        # Unpack the pre-split data from the calling script
        m_train_x, m_train_y, m_valid_x, m_valid_y, m_train_meta, m_valid_meta = manual_split
        
        # Derive person mappings from the metadata
        unique_p = sorted(set([m['person'] for m in m_train_meta]))
        person_to_id = {p: i for i, p in enumerate(unique_p)}
        id_to_person = {i: p for p, i in person_to_id.items()}
        
        # Override to ensure we only run the provided split (as 'Fold 0')
        params["folds"] = [0]
    else:
        # Standard workflow: Load full dataset from disk
        x_all, y_all, metadata, person_to_id, id_to_person, action_names = data_loading.load_person_dataset(
            sensor=params["sensor_data"],
            version=params["version"],
            action_indices=params["action_indices"],
            data_dir=params["data_dir"],
            window_len=params["window_len"],
            stride=params["stride"],
        )
        # x_all = data_loading.normalize(x_all, metadata) causing leakage
        fold_indices = data_loading.make_stratified_folds(
            y_all, n_splits=5, seed=params["seed"]
        )

    results = []
    aggregate_true = []
    aggregate_pred = []
    experiment_name = params["experiment_name"] or f"sensor_{params['sensor_data'].lower()}"
    experiment_dir = None
    if params["save_results"]:
        experiment_dir = eval_utils.resolve_results_dir(
            task_name="person_identification",
            experiment_name=experiment_name,
            results_root=params["results_root"],
        )

    for fold in params["folds"]:
        print(f"\n===== FOLD {fold} =====")

        if manual_split is not None:
            train_x, train_y = m_train_x, m_train_y
            valid_x, valid_y = m_valid_x, m_valid_y
            train_meta, valid_meta = m_train_meta, m_valid_meta
        else:
            train_x, train_y, valid_x, valid_y, train_meta, valid_meta = data_loading.get_fold_split(
                x_all, y_all, metadata, fold_indices, fold=fold
            )



        if params["print_summary"]:
            data_loading.print_dataset_summary(
                train_x, train_y, train_meta, id_to_person, title=f"train fold {fold}"
            )
            data_loading.print_dataset_summary(
                valid_x, valid_y, valid_meta, id_to_person, title=f"valid fold {fold}"
            )

        # Preprocessing for Torch
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
            shuffle=True,
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=params["bsize"],
            shuffle=False,
            drop_last=False,
        )

        input_dim = train_x_pad.shape[2]
        num_classes = len(person_to_id)

        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=params["dropout"],
            num_classes=num_classes,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

        label_names = [id_to_person[i] for i in range(num_classes)]
        best_valid_acc = -1.0
        best_epoch = -1
        best_valid_loss = float("inf")
        patience_counter = 0
        history = []

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
                    "task": "person_identification",
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
            [id_to_person[i] for i in range(len(id_to_person))],
            experiment_dir / "aggregate",
            metadata={
                "task": "person_identification",
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
    train_person_lstm()


if __name__ == "__main__":
    main()
