import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import LSTM.data_loading_person_split_yichiao as data_loading

class SequenceDataset(Dataset):
    def __init__(self, x, y, l):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.l = torch.tensor(l, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i], self.l[i]

class LSTMClassifier(nn.Module):
    def __init__(self, in_d, hid_d, n_lay, drop, n_cl):
        super().__init__()
        self.lstm = nn.LSTM(in_d, hid_d, n_lay, batch_first=True, dropout=(drop if n_lay > 1 else 0))
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hid_d, n_cl)
    def forward(self, x, l):
        packed = nn.utils.rnn.pack_padded_sequence(x, l.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.dropout(h[-1]))

def train_person_lstm(params, preloaded_data=None):
    device = torch.device(params.get("device", "cpu"))
    # Expecting: (train_x, train_y, valid_x, valid_y, train_meta, valid_meta)
    tx, ty, vx, vy, tm, vm = preloaded_data
    
    unique_p = sorted(set([m['person'] for m in tm]))
    id2p = {i: p for i, p in enumerate(unique_p)}
    num_cl = len(unique_p)

    if params.get("print_summary", True):
        data_loading.print_dataset_summary(tx, ty, tm, id2p, "Train (Session 1)")
        data_loading.print_dataset_summary(vx, vy, vm, id2p, "Valid (Session 2)")

    tx_p, ty_p, tl = data_loading.pad_for_torch(tx, ty)
    vx_p, vy_p, vl = data_loading.pad_for_torch(vx, vy, max_len=tx_p.shape[1])

    train_loader = DataLoader(SequenceDataset(tx_p, ty_p, tl), batch_size=params["bsize"], shuffle=True)
    valid_loader = DataLoader(SequenceDataset(vx_p, vy_p, vl), batch_size=params["bsize"])

    model = LSTMClassifier(tx_p.shape[2], params["lstm_layers"][0], 1, params.get("dropout", 0.5), num_cl).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 1e-4))
    crit = nn.CrossEntropyLoss()

    for e in range(params["nepochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {e+1}/{params['nepochs']}")
        for bx, by, bl in pbar:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(); loss = crit(model(bx, bl), by); loss.backward(); opt.step()
            pbar.set_postfix(loss=loss.item())
        
        model.eval(); ok = 0
        with torch.no_grad():
            for bx, by, bl in valid_loader:
                ok += (model(bx.to(device), bl).argmax(1) == by.to(device)).sum().item()
        print(f" >> Epoch {e+1} | Valid Acc: {ok/len(vy):.4f}")