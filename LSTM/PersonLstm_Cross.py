"""
PyTorch LSTM for person identification with explicit cross-session splitting.

Train on version 1 files and evaluate on version 2 files.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import LSTM.data_loading_person_split_yichiao as data_loading
    import LSTM.eval_utils_yichiao as eval_utils
except ModuleNotFoundError:
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
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]
        return self.fc(self.dropout(last_hidden))


def _parse_lstm_layers(lstm_layers):
    if not lstm_layers:
        raise ValueError("lstm_layers cannot be empty")
    hidden_dim = lstm_layers[0]
    num_layers = len(lstm_layers)
    return hidden_dim, num_layers


def _normalize_with_train_stats(train_x, test_x):
    train_flat = np.concatenate(train_x, axis=0)
    mu = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std == 0] = 1.0
    train_x = np.array([(sample - mu) / std for sample in train_x], dtype=object)
    test_x = np.array([(sample - mu) / std for sample in test_x], dtype=object)
    return train_x, test_x


def load_cross_session_split(sensor="all", action_indices=None, data_dir=None):
    train_x, train_y, train_meta, person_to_id, id_to_person, _ = data_loading.load_person_dataset(
        sensor=sensor,
        version=1,
        action_indices=action_indices,
        data_dir=data_dir,
    )
    test_x, test_y, test_meta, _, _, _ = data_loading.load_person_dataset(
        sensor=sensor,
        version=2,
        action_indices=action_indices,
        data_dir=data_dir,
        person_to_id=person_to_id,
    )

    if len(train_x) == 0:
        raise ValueError("No training samples were loaded from version 1 files")
    if len(test_x) == 0:
        raise ValueError("No test samples were loaded from version 2 files")

    train_people = {meta["person"] for meta in train_meta}
    test_people = {meta["person"] for meta in test_meta}
    if train_people != test_people:
        raise ValueError(
            f"Train/test person mismatch. train={sorted(train_people)} test={sorted(test_people)}"
        )

    return train_x, train_y, train_meta, test_x, test_y, test_meta, person_to_id, id_to_person


def train_one_epoch(model, loader, optimizer, criterion, device):
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
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
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
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_count += y_batch.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for x_batch, y_batch, lengths in loader:
        logits = model(x_batch.to(device), lengths.to(device))
        y_true.extend(y_batch.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    return y_true, y_pred


def train_person_lstm_cross_session(user_params=None):
    params = {
        "sensor_data": "all",
        "action_indices": list(range(21)),
        "lstm_layers": [400],
        "nepochs": 5,
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
        "experiment_name": "CrossSession_Person_LSTM",
    }

    if user_params:
        params.update(user_params)

    set_seed(params["seed"])
    device = torch.device(params["device"])
    hidden_dim, num_layers = _parse_lstm_layers(params["lstm_layers"])

    train_x, train_y, train_meta, test_x, test_y, test_meta, person_to_id, id_to_person = (
        load_cross_session_split(
            sensor=params["sensor_data"],
            action_indices=params["action_indices"],
            data_dir=params["data_dir"],
        )
    )

    train_x, test_x = _normalize_with_train_stats(train_x, test_x)

    if params["print_summary"]:
        data_loading.print_dataset_summary(train_x, train_y, train_meta, id_to_person, title="Train Session (_1)")
        data_loading.print_dataset_summary(test_x, test_y, test_meta, id_to_person, title="Test Session (_2)")

    train_x_pad, train_y_out, train_lengths = data_loading.pad_for_torch(
        train_x,
        train_y,
        max_len=params["max_len"],
    )
    pad_max_len = params["max_len"] if params["max_len"] is not None else train_x_pad.shape[1]
    test_x_pad, test_y_out, test_lengths = data_loading.pad_for_torch(
        test_x,
        test_y,
        max_len=pad_max_len,
    )

    train_loader = DataLoader(
        SequenceDataset(train_x_pad, train_y_out, train_lengths),
        batch_size=params["bsize"],
        shuffle=True,
    )
    test_loader = DataLoader(
        SequenceDataset(test_x_pad, test_y_out, test_lengths),
        batch_size=params["bsize"],
        shuffle=False,
    )

    model = LSTMClassifier(
        input_dim=train_x_pad.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=params["dropout"],
        num_classes=len(person_to_id),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    history = []
    best_test_acc = -1.0
    best_epoch = -1

    for epoch in range(params["nepochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
            }
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

        print(
            f"Epoch {epoch + 1}/{params['nepochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    test_true, test_pred = collect_predictions(model, test_loader, device)
    label_names = [id_to_person[i] for i in range(len(person_to_id))]

    result = {
        "best_test_acc": float(best_test_acc),
        "best_epoch": best_epoch,
        "history": history,
        "test_true": test_true,
        "test_pred": test_pred,
        "label_names": label_names,
    }

    if params["save_results"]:
        experiment_dir = eval_utils.resolve_results_dir(
            task_name="person_identification",
            experiment_name=params["experiment_name"],
            results_root=params["results_root"],
        )
        eval_utils.save_history(history, experiment_dir)
        report = eval_utils.save_classification_report(
            test_true,
            test_pred,
            label_names,
            experiment_dir,
            metadata={
                "task": "person_identification",
                "split": "cross_session",
                "train_version": 1,
                "test_version": 2,
                "sensor": params["sensor_data"],
                "params": params,
            },
        )
        result["report_summary_path"] = report["summary_path"]

    print("\nDone.")
    print({"best_test_acc": result["best_test_acc"], "best_epoch": result["best_epoch"]})
    return result


def main():
    train_person_lstm_cross_session()


if __name__ == "__main__":
    main()
