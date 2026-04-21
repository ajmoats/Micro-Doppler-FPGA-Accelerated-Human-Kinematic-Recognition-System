"""
PyTorch LSTM for person identification.
EMERGENCY VERSION: Bypasses broken sklearn/joblib by using manual NumPy logic.
Includes Matplotlib-based Viridis confusion matrix and per-person accuracy reporting.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
    """Prints a text report using manually calculated accuracy per ID."""
    print("\n--- Per-Person Accuracy (Cross-Session) ---")
    unique_ids = sorted(id_to_person.keys())
    for p_id in unique_ids:
        indices = [i for i, label in enumerate(all_labels) if label == p_id]
        if not indices:
            continue
        correct = sum(1 for i in indices if all_preds[i] == all_labels[i])
        total = len(indices)
        acc = (correct / total) * 100
        print(f"{id_to_person[p_id]:<10}: {acc:>6.2f}% ({correct}/{total})")
    print("-------------------------------------------\n")

def plot_confusion_matrix(all_labels, all_preds, id_to_person, title="Confusion Matrix"):
    """Manual Confusion Matrix calculation and Viridis plotting (No Sklearn)."""
    unique_ids = sorted(id_to_person.keys())
    person_names = [id_to_person[i] for i in unique_ids]
    n_classes = len(unique_ids)
    
    # 1. Manual CM Calculation
    cm = np.zeros((n_classes, n_classes), dtype=int)
    id_map = {id_val: i for i, id_val in enumerate(unique_ids)}
    for true_val, pred_val in zip(all_labels, all_preds):
        if true_val in id_map and pred_val in id_map:
            cm[id_map[true_val], id_map[pred_val]] += 1
    
    # 2. Manual Normalization
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), row_sums, 
                        out=np.zeros_like(cm.astype('float')), where=row_sums!=0)

    # 3. Matplotlib Heatmap Rendering
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='viridis')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Match Probability", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(person_names)))
    ax.set_yticks(np.arange(len(person_names)))
    ax.set_xticklabels(person_names)
    ax.set_yticklabels(person_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(len(person_names)):
        for j in range(len(person_names)):
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    ax.set_title(f"{title}\nColorblind-Friendly (Viridis)")
    ax.set_ylabel('True Person (Session 2)')
    ax.set_xlabel('Predicted Person (Session 1 Model)')
    fig.tight_layout()
    plt.show()

def train_person_lstm(user_params=None, preloaded_data=None):
    params = {
        "sensor_data": "all",
        "nepochs": 5,
        "bsize": 16,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "print_summary": True,
    }
    if user_params: params.update(user_params)
    set_seed(1337)
    device = torch.device(params["device"])
    
    # Hidden dim/layers parsing (hardcoded for baseline if simple)
    hidden_dim = 400
    num_layers = 1

    # --- DATA HANDLING ---
    m_train_x, m_train_y, m_valid_x, m_valid_y, m_train_meta, m_valid_meta = preloaded_data
    unique_p = sorted(set([m['person'] for m in m_train_meta]))
    num_classes = len(unique_p)
    id_to_person = {i: p for i, p in enumerate(unique_p)}

    # Padding
    train_x_pad, train_y_out, train_len = data_loading.pad_for_torch(m_train_x, m_train_y)
    valid_x_pad, valid_y_out, valid_len = data_loading.pad_for_torch(m_valid_x, m_valid_y, max_len=train_x_pad.shape[1])

    train_loader = DataLoader(SequenceDataset(train_x_pad, train_y_out, train_len), batch_size=params["bsize"], shuffle=True)
    valid_loader = DataLoader(SequenceDataset(valid_x_pad, valid_y_out, valid_len), batch_size=params["bsize"])

    # Init Model
    model = LSTMClassifier(train_x_pad.shape[2], hidden_dim, num_layers, 0.5, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    for epoch in range(params["nepochs"]):
        _, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        _, v_acc = evaluate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1} | train_acc={t_acc:.4f} valid_acc={v_acc:.4f}")

    # --- FINAL BREAKDOWN ---
    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch, lengths in valid_loader:
            x_batch, y_batch, lengths = x_batch.to(device), y_batch.to(device), lengths.to(device)
            logits = model(x_batch, lengths)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_actuals.extend(y_batch.cpu().numpy())
    
    print_per_person_accuracy(all_actuals, all_preds, id_to_person)
    plot_confusion_matrix(all_actuals, all_preds, id_to_person, title="Cross-Session Person ID Result")

    return [{"fold": 0, "best_valid_acc": v_acc}]