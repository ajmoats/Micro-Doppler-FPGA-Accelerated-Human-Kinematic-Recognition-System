"""
PyTorch LSTM for person identification.
EMERGENCY VERSION: Synchronized for Parallel Execution.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm  # for progress bars in terminal
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

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx], self.lengths[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, num_layers=1, dropout=0.5, num_classes=10):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


def _parse_lstm_layers(lstm_layers):
    hidden_dim = lstm_layers[0]
    num_layers = len(lstm_layers)
    return hidden_dim, num_layers


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for x_batch, y_batch, lengths in pbar:
        x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x_batch, lengths)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_count += y_batch.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x_batch, y_batch, lengths in loader:
        x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)
        logits = model(x_batch, lengths)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_count += y_batch.size(0)
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Collect all predictions and ground truth labels from a DataLoader."""
    model.eval()
    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in loader:
        logits = model(x_batch.to(device), lengths.to(device))
        y_true.extend(y_batch.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return y_true, y_pred


def train_person_lstm(user_params=None, preloaded_data=None, run_id="default"):
    params = {
        "sensor_data": "US40",
        "version": 1,
        "action_indices": list(range(21)),
        "lstm_layers": [400],
        "nepochs": 1,
        "folds": [0],
        "seed": 1337,
        "dropout": 0.5,
        "bsize": 16,
        "max_len": None,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "mask_val": 0.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": None,
        "print_summary": True,
        "save_results": True,
        "results_root": None,
        "experiment_name": "Split_Person_LSTM",
    }

    if user_params:
        params.update(user_params)
    set_seed(params["seed"])
    device = torch.device(params["device"])
    hidden_dim, num_layers = _parse_lstm_layers(params["lstm_layers"])

    # --- DATA HANDLING ---
    if preloaded_data is not None:
        print(">>> Using preloaded data split")
        if len(preloaded_data) == 4:
            m_train_x, m_train_y, m_valid_x, m_valid_y = preloaded_data
            m_train_meta, m_valid_meta = None, None
            all_labels = np.concatenate([m_train_y, m_valid_y])
            unique_classes = np.unique(all_labels)
            num_classes = int(unique_classes.max() + 1)
            id_to_person = {i: f"ID_{i}" for i in unique_classes}
        else:
            m_train_x, m_train_y, m_valid_x, m_valid_y, m_train_meta, m_valid_meta = preloaded_data
            unique_p = sorted(set([m['person'] for m in m_train_meta]))
            num_classes = len(unique_p)
            id_to_person = {i: p for i, p in enumerate(unique_p)}
        params["folds"] = [0]
    else:
        x_all, y_all, metadata, p2id, id_to_person, _ = data_loading.load_person_dataset(
            sensor=params["sensor_data"], version=params["version"], data_dir=params["data_dir"]
        )
        num_classes = len(p2id)
        # NOTE: normalize is intentionally NOT called here to prevent leakage.
        # Normalization is done per-fold after splitting (see below).
        fold_indices = data_loading.make_stratified_folds(y_all, n_splits=5, seed=params["seed"])

    # --- RESULTS DIR SETUP ---
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

    label_names = [id_to_person[i] for i in range(num_classes)]

    # --- FOLD LOOP ---
    for fold in params["folds"]:
        print(f"\n===== FOLD {fold} =====")

        if preloaded_data is not None:
            train_x, train_y = m_train_x, m_train_y
            valid_x, valid_y = m_valid_x, m_valid_y
            train_meta, valid_meta = m_train_meta, m_valid_meta
        else:
            train_x, train_y, valid_x, valid_y, train_meta, valid_meta = data_loading.get_fold_split(
                x_all, y_all, metadata, fold_indices, fold=fold
            )

        # Normalize using training data stats only — prevents leakage into validation set
        train_flat = np.concatenate(train_x, axis=0)
        mu = train_flat.mean(axis=0)
        std = train_flat.std(axis=0)
        std[std == 0] = 1.0
        train_x = np.array([(s - mu) / std for s in train_x], dtype=object)
        valid_x = np.array([(s - mu) / std for s in valid_x], dtype=object)

        if params["print_summary"] and hasattr(data_loading, 'print_dataset_summary'):
            data_loading.print_dataset_summary(train_x, train_y, train_meta, id_to_person, title=f"Train Fold {fold}")
            data_loading.print_dataset_summary(valid_x, valid_y, valid_meta, id_to_person, title=f"Valid Fold {fold}")

        train_x_pad, train_y_out, train_len = data_loading.pad_for_torch(train_x, train_y)
        valid_x_pad, valid_y_out, valid_len = data_loading.pad_for_torch(valid_x, valid_y, max_len=train_x_pad.shape[1])

        train_loader = DataLoader(
            SequenceDataset(train_x_pad, train_y_out, train_len),
            batch_size=params["bsize"],
            shuffle=True,
        )
        valid_loader = DataLoader(
            SequenceDataset(valid_x_pad, valid_y_out, valid_len),
            batch_size=params["bsize"],
            shuffle=False,
        )

        model = LSTMClassifier(
            train_x_pad.shape[2], hidden_dim, num_layers, params["dropout"], num_classes
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )

        best_acc = -1.0
        best_epoch = -1
        history = []

        for epoch in range(params["nepochs"]):
            t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            v_loss, v_acc = evaluate(model, valid_loader, criterion, device)

            history.append({
                "epoch": epoch + 1,
                "train_loss": float(t_loss),
                "train_acc": float(t_acc),
                "valid_loss": float(v_loss),
                "valid_acc": float(v_acc),
            })

            if v_acc > best_acc:
                best_acc = v_acc
                best_epoch = epoch + 1

            print(f"Fold {fold} | Epoch {epoch+1}/{params['nepochs']} | "
                  f"train_loss={t_loss:.4f} train_acc={t_acc:.4f} | "
                  f"valid_loss={v_loss:.4f} valid_acc={v_acc:.4f}")

        # --- COLLECT PREDICTIONS FOR THIS FOLD ---
        fold_true, fold_pred = collect_predictions(model, valid_loader, device)
        aggregate_true.extend(fold_true)
        aggregate_pred.extend(fold_pred)

        fold_result = {
            "fold": fold,
            "best_valid_acc": best_acc,
            "best_epoch": best_epoch,
            "history": history,
            "valid_true": fold_true,
            "valid_pred": fold_pred,
        }

        # --- SAVE PER-FOLD RESULTS ---
        if experiment_dir is not None:
            fold_dir = experiment_dir / f"fold_{fold}"
            eval_utils.save_history(history, fold_dir)
            eval_utils.save_classification_report(
                fold_true,
                fold_pred,
                label_names,
                fold_dir,
                metadata={
                    "task": "person_identification",
                    "sensor": params["sensor_data"],
                    "fold": fold,
                    "params": params,
                },
            )

        results.append(fold_result)

    # --- SAVE AGGREGATE RESULTS ACROSS ALL FOLDS ---
    if experiment_dir is not None and aggregate_true:
        eval_utils.save_classification_report(
            aggregate_true,
            aggregate_pred,
            label_names,
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
        print({"fold": r["fold"], "best_valid_acc": r["best_valid_acc"], "best_epoch": r["best_epoch"]})

    return results

if __name__ == "__main__":
    params = {
        "sensor_data": "all",
        "version": 1,
        "action_indices": list(range(21)) ,
        "lstm_layers": [400],
        "nepochs": 5,
        "folds": list(range(5)),
        "seed": 1337,
        "dropout": 0.5,
        "bsize": 16,
        "max_len": None,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "mask_val": 0.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": None,
        "print_summary": True,
        "save_results": True,
        "results_root": "results",
        "experiment_name": "Split_Person_LSTM_ALL",
    }

    train_person_lstm(user_params=params)