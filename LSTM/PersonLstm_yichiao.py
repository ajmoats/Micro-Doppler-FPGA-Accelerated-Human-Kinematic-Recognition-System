"""
PyTorch LSTM for person identification.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import data_loading_person_yichiao as data_loading


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


def train_person_lstm(user_params=None):
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
        "mask_val": 0.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": None,
        "print_summary": True,
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

    # Load dataset
    x_all, y_all, metadata, person_to_id, id_to_person, action_names = data_loading.load_person_dataset(
        sensor=params["sensor_data"],
        version=params["version"],
        action_indices=params["action_indices"],
        data_dir=params["data_dir"],
    )

    x_all = data_loading.normalize(x_all, metadata)
    fold_indices = data_loading.make_stratified_folds(
        y_all, n_splits=5, seed=params["seed"]
    )

    results = []

    for fold in params["folds"]:
        print(f"\n===== FOLD {fold} =====")

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

        best_valid_acc = -1.0
        best_epoch = -1

        for epoch in range(params["nepochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            valid_loss, valid_acc = evaluate(
                model, valid_loader, criterion, device
            )

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch + 1

            print(
                f"Epoch {epoch + 1}/{params['nepochs']} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f}"
            )

        fold_result = {
            "fold": fold,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "best_valid_acc": best_valid_acc,
            "best_epoch": best_epoch,
        }
        results.append(fold_result)

    print("\nDone.")
    print("Fold results:")
    for r in results:
        print(r)

    return results


def main():
    train_person_lstm()


if __name__ == "__main__":
    main()