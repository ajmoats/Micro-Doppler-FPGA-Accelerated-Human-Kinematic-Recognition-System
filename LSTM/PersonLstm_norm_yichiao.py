"""
PyTorch LSTM for person identification.
Updated for numerical stability (Gradient Clipping) and Source-Separation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import data_loading_person_yichiao as data_loading


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, x, y, lengths):
        # Keep as tensors for speed
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
    hidden_dim = lstm_layers[0]
    num_layers = len(lstm_layers)
    return hidden_dim, num_layers


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    
    # Using tqdm for visibility as it helps track those 6-10 min epochs
    pbar = tqdm(loader, desc="Training", leave=False)

    for x_batch, y_batch, lengths in pbar:
        x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)

        optimizer.zero_grad()
        logits = model(x_batch, lengths)
        loss = criterion(logits, y_batch)
        
        if torch.isnan(loss):
            continue

        loss.backward()
        
        # --- STABILITY FIX: Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_batch).sum().item()
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
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_count += y_batch.size(0)
    return total_loss / total_count, total_correct / total_count


def train_person_lstm(user_params=None, manual_split=None):
    params = {
        "sensor_data": "US40",
        "version": 1,
        "action_indices": list(range(21)),
        "lstm_layers": [400],
        "nepochs": 50,
        "patience": 5,
        "folds": [0],
        "seed": 1337,
        "dropout": 0.5,
        "bsize": 16,
        "max_len": None, # If None, pad_for_torch uses max of current fold
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "mask_val": 0.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": None,
        "print_summary": True,
        "window_len": None,
        "stride": None,
    }

    if user_params:
        params.update(user_params)

    set_seed(params["seed"])
    device = torch.device(params["device"])
    hidden_dim, num_layers = _parse_lstm_layers(params["lstm_layers"])

    # --- DATA HANDLING ---
    if manual_split is not None:
        m_train_x, m_train_y, m_valid_x, m_valid_y, m_train_meta, m_valid_meta = manual_split
        unique_p = sorted(set([m['person'] for m in m_train_meta]))
        person_to_id = {p: i for i, p in enumerate(unique_p)}
        id_to_person = {i: p for p, i in person_to_id.items()}
        params["folds"] = [0]
    else:
        x_all, y_all, metadata, person_to_id, id_to_person, action_names = data_loading.load_person_dataset(
            sensor=params["sensor_data"], version=params["version"],
            action_indices=params["action_indices"], data_dir=params["data_dir"]
        )
        x_all = data_loading.normalize(x_all, metadata)
        # Use updated fold logic (metadata-based)
        fold_indices = data_loading.make_stratified_folds(metadata, n_splits=5, seed=params["seed"])

    results = []

    for fold in params["folds"]:
        if manual_split is not None:
            train_x, train_y, valid_x, valid_y = m_train_x, m_train_y, m_valid_x, m_valid_y
            train_meta, valid_meta = m_train_meta, m_valid_meta
        else:
            train_x, train_y, valid_x, valid_y, train_meta, valid_meta = data_loading.get_fold_split(
                x_all, y_all, metadata, fold_indices, fold=fold
            )

        train_x_pad, train_y_out, train_lengths = data_loading.pad_for_torch(train_x, train_y, max_len=params["max_len"])
        valid_x_pad, valid_y_out, valid_lengths = data_loading.pad_for_torch(valid_x, valid_y, max_len=params["max_len"])

        train_ds = SequenceDataset(train_x_pad, train_y_out, train_lengths)
        valid_ds = SequenceDataset(valid_x_pad, valid_y_out, valid_lengths)

        train_loader = DataLoader(train_ds, batch_size=params["bsize"], shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=params["bsize"], shuffle=False)

        model = LSTMClassifier(train_x_pad.shape[2], hidden_dim, num_layers, params["dropout"], len(person_to_id)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        criterion = nn.CrossEntropyLoss()

        best_valid_loss = float('inf')
        best_valid_acc = -1.0
        best_epoch = -1
        trigger_times = 0

        for epoch in range(params["nepochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch + 1

            print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f} | Valid Loss: {valid_loss:.4f}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= params["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        results.append({"fold": fold, "best_valid_acc": best_valid_acc, "best_epoch": best_epoch})

    return results

if __name__ == "__main__":
    train_person_lstm()