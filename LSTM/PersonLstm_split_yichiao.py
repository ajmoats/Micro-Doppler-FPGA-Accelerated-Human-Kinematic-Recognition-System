"""
PyTorch LSTM for person identification.
EMERGENCY VERSION: Synchronized for Parallel Execution.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
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
        return self.fc(self.dropout(h_n[-1]))

def plot_confusion_matrix(all_labels, all_preds, id_to_person, title, run_id):
    unique_ids = sorted(id_to_person.keys())
    n_classes = len(unique_ids)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    id_map = {id_val: i for i, id_val in enumerate(unique_ids)}
    
    for true_val, pred_val in zip(all_labels, all_preds):
        cm[id_map[true_val], id_map[pred_val]] += 1
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap='viridis')
    ax.set_title(title)
    plt.colorbar(im)
    
    # Save using the unique run_id to prevent parallel overwrite
    plt.savefig(f'confusion_matrix_{run_id}.png', dpi=300) 
    plt.close()

def train_person_lstm(user_params=None, preloaded_data=None, run_id="default"):
    params = {
        "nepochs": 5, "bsize": 16, "lr": 1e-4, "weight_decay": 1e-4,
        "device": "cpu", "seed": 1337
    }
    if user_params: params.update(user_params)
    
    # CRITICAL: Use the seed passed in from test_run
    set_seed(params["seed"])
    device = torch.device(params["device"])
    
    tx, ty, vx, vy, tm, vm = preloaded_data
    unique_p = sorted(set([m['person'] for m in tm]))
    id_to_person = {i: p for i, p in enumerate(unique_p)}

    train_x_pad, train_y_out, train_len = data_loading.pad_for_torch(tx, ty)
    valid_x_pad, valid_y_out, valid_len = data_loading.pad_for_torch(vx, vy, max_len=train_x_pad.shape[1])

    train_loader = DataLoader(SequenceDataset(train_x_pad, train_y_out, train_len), batch_size=params["bsize"], shuffle=True)
    valid_loader = DataLoader(SequenceDataset(valid_x_pad, valid_y_out, valid_len), batch_size=params["bsize"])

    model = LSTMClassifier(train_x_pad.shape[2], 400, 1, 0.5, len(unique_p)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    for epoch in range(params["nepochs"]):
        model.train()
        for xb, yb, le in train_loader:
            xb, yb, le = xb.to(device), yb.to(device), le.to(device)
            optimizer.zero_grad(); loss = criterion(model(xb, le), yb); loss.backward(); optimizer.step()

    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for xb, yb, le in valid_loader:
            xb, yb, le = xb.to(device), yb.to(device), le.to(device)
            all_preds.extend(model(xb, le).argmax(dim=1).cpu().numpy())
            all_actuals.extend(yb.cpu().numpy())
    
    plot_confusion_matrix(all_actuals, all_preds, id_to_person, f"Result: {run_id}", run_id)
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_actuals))
    return accuracy