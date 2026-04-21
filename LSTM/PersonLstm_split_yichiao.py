"""
PyTorch LSTM for person identification.
Updated to support manual data splits for Source-Separation and Cross-Session testing.
Includes colorblind-friendly Viridis confusion matrix and per-person accuracy reporting.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader

# Ensure this matches the name of your data loading script
import data_loading_person_split_yichiao as data_loading

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
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
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
    for x_batch, y_batch, lengths in loader:
        x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)
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
    total_loss, total_correct, total_count = 0.0, 0, 0
    for x_batch, y_batch, lengths in loader:
        x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)
        logits = model(x_batch, lengths)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * y_batch.size(0)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_count += y_batch.size(0)
    return total_loss / total_count, total_correct / total_count

def print_per_person_accuracy(all_labels, all_preds, id_to_person):
    """Prints a detailed text report of accuracy for each participant."""
    print("\n--- Per-Person Accuracy (Cross-Session) ---")
    unique_ids = sorted(id_to_person.keys())
    for p_id in unique_ids:
        indices = [i for i, label in enumerate(all_labels) if label == p_id]
        if len(indices) == 0:
            print(f"{id_to_person[p_id]:<10}: No samples in test set")
            continue
        correct = sum(1 for i in indices if all_preds[i] == all_labels[i])
        total = len(indices)
        acc = (correct / total) * 100
        print(f"{id_to_person[p_id]:<10}: {acc:>6.2f}% ({correct}/{total})")
    print("-------------------------------------------\n")

def plot_confusion_matrix(all_labels, all_preds, id_to_person, title="Confusion Matrix"):
    """Generates a colorblind-friendly heatmap using the Viridis scheme."""
    unique_ids = sorted(id_to_person.keys())
    person_names = [id_to_person[i] for i in unique_ids]
    cm = confusion_matrix(all_labels, all_preds, labels=unique_ids)
    
    # Normalize by row (True Labels)
    cm_norm = np.divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis], 
                        out=np.zeros_like(cm.astype('float')), where=cm.sum(axis=1)[:, np.newaxis]!=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='viridis', 
                xticklabels=person_names, yticklabels=person_names,
                cbar_kws={'label': 'Identification Probability'})
    
    plt.title(f"{title}\nColorblind-Sensitive (Viridis)", fontsize=15)
    plt.ylabel('True Person (Session 2)', fontsize=12)
    plt.xlabel('Predicted Person (Session 1 Model)', fontsize=12)
    plt.tight_layout()
    plt.show()

def train_person_lstm(user_params=None, preloaded_data=None):
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
    }
    if user_params: params.update(user_params)
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
        x_all = data_loading.normalize(x_all, metadata)
        fold_indices = data_loading.make_stratified_folds(y_all, n_splits=5, seed=params["seed"])

    results = []
    for fold in params["folds"]:
        if preloaded_data is not None:
            train_x, train_y = m_train_x, m_train_y
            valid_x, valid_y = m_valid_x, m_valid_y
            train_meta, valid_meta = m_train_meta, m_valid_meta
        else:
            train_x, train_y, valid_x, valid_y, train_meta, valid_meta = data_loading.get_fold_split(
                x_all, y_all, metadata, fold_indices, fold=fold
            )

        if params["print_summary"] and hasattr(data_loading, 'print_dataset_summary'):
            data_loading.print_dataset_summary(train_x, train_y, train_meta, id_to_person, title=f"Fold {fold}")

        train_x_pad, train_y_out, train_len = data_loading.pad_for_torch(train_x, train_y)
        valid_x_pad, valid_y_out, valid_len = data_loading.pad_for_torch(valid_x, valid_y, max_len=train_x_pad.shape[1])

        train_loader = DataLoader(SequenceDataset(train_x_pad, train_y_out, train_len), batch_size=params["bsize"], shuffle=True)
        valid_loader = DataLoader(SequenceDataset(valid_x_pad, valid_y_out, valid_len), batch_size=params["bsize"])

        model = LSTMClassifier(train_x_pad.shape[2], hidden_dim, num_layers, params["dropout"], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

        best_acc = -1.0
        for epoch in range(params["nepochs"]):
            _, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            _, v_acc = evaluate(model, valid_loader, criterion, device)
            best_acc = max(best_acc, v_acc)
            print(f"Fold {fold} | Epoch {epoch+1} | train_acc={t_acc:.4f} valid_acc={v_acc:.4f}")

        # --- FINAL ANALYSIS AFTER TRAINING ---
        model.eval()
        all_preds, all_actuals = [], []
        with torch.no_grad():
            for x_batch, y_batch, lengths in valid_loader:
                x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)
                logits = model(x_batch, lengths)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_actuals.extend(y_batch.cpu().numpy())
        
        # 1. Print Text Report
        print_per_person_accuracy(all_actuals, all_preds, id_to_person)
        
        # 2. Display Heatmap
        plot_confusion_matrix(all_actuals, all_preds, id_to_person, 
                              title="Cross-Session Person ID (V1 Train -> V2 Test)")

        results.append({"fold": fold, "best_valid_acc": best_acc})
    
    return results