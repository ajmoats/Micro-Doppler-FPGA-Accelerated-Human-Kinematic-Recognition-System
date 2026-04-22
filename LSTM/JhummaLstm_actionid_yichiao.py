"""
JhummaLstm_actionid_yichiao.py
Complete PyTorch LSTM for Action Recognition with Per-Actor Breakdown.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import LSTM.data_loading_actionid_yichiao as data_loading

# --- REPRODUCIBILITY ---
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed_all(seed)

# --- DATASET OBJECT ---
class SequenceDataset(Dataset):
    def __init__(self, x, y, lengths, actors):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.l = torch.tensor(lengths, dtype=torch.long)
        self.a = torch.tensor(actors, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx], self.l[idx], self.a[idx]

# --- MODEL DEFINITION ---
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, num_classes=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.fc(self.dropout(h_n[-1]))

# --- MAIN TRAINING WRAPPER ---
def train_lstm(user_params=None):
    params = {
        "seed": 1337,
        "device": "cpu",
        "sensor_data": "all",
        "folds": [0],
        "nepochs": 5,
        "bsize": 50,
        "lr": 1e-3,
        "max_len": 404
    }
    if user_params: params.update(user_params)

    set_seed(params["seed"])
    device = torch.device(params["device"])

    print(f"Loading data for sensor: {params['sensor_data']}...")
    # FIXED: Now expects 4 values
    x_all, y_all, batch_ids, actor_ids = data_loading.load_sensor_data(sensor=params["sensor_data"])
    x_all = data_loading.normalize(x_all)

    for fold in params["folds"]:
        print(f"\n===== STARTING FOLD {fold} =====")
        train_x, train_y, train_act, valid_x, valid_y, valid_act = \
            data_loading.split_5_folds_with_actors(x_all, y_all, batch_ids, actor_ids, fold=fold)

        tx_p, ty, tl = data_loading.pad_for_torch(train_x, train_y, max_len=params["max_len"])
        vx_p, vy, vl = data_loading.pad_for_torch(valid_x, valid_y, max_len=params["max_len"])

        t_loader = DataLoader(SequenceDataset(tx_p, ty, tl, train_act), batch_size=params["bsize"], shuffle=True)
        v_loader = DataLoader(SequenceDataset(vx_p, vy, vl, valid_act), batch_size=params["bsize"], shuffle=False)

        model = LSTMClassifier(tx_p.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(params["nepochs"]):
            model.train()
            for bx, by, bl, _ in t_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(); criterion(model(bx, bl), by).backward(); optimizer.step()

            model.eval(); v_ok = 0
            with torch.no_grad():
                for bx, by, bl, _ in v_loader:
                    v_ok += (model(bx.to(device), bl).argmax(1) == by.to(device)).sum().item()
            print(f"Epoch {epoch+1}/{params['nepochs']} | Valid Acc: {v_ok/len(vy):.4f}")

        # FINAL ACTOR BREAKDOWN
        actor_stats = {i: {"c": 0, "t": 0} for i in range(13)}
        model.eval()
        with torch.no_grad():
            for bx, by, bl, ba in v_loader:
                preds = model(bx.to(device), bl).argmax(1)
                for p, g, a in zip(preds, by, ba):
                    a_id = a.item()
                    actor_stats[a_id]["t"] += 1
                    if p == g.to(device): actor_stats[a_id]["c"] += 1
        
        print(f"\n--- Fold {fold} Participant Breakdown ---")
        for i in range(13):
            s = actor_stats[i]
            if s["t"] > 0: print(f"Actor {i:02d}: {s['c']/s['t']:.2%} ({s['c']}/{s['t']})")

if __name__ == "__main__":
    train_lstm()