"""
data_loading_actionid_yichiao.py
Modified to return (stacked_data, labels, batch_ids, actor_ids)
"""

from pathlib import Path
import numpy as np
import scipy.io as sio

CHUNKS, ACTORS = 10, 13
KEEP_ACTIONS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

def resolve_data_path(custom_path=None):
    repo_root = Path(__file__).resolve().parent.parent
    p = repo_root / "data" / "data_rot_chunks10.mat"
    return p

def _load_one_sensor_from_raw(raw_data, sensor="US40"):
    sensor_raw = raw_data["dataChunks" + sensor]
    rand_perm_idx = raw_data["randpermIx"] - 1
    data_list, labels_list, batch_ids_list, actor_ids_list = [], [], [], []

    for chunk in range(CHUNKS):
        for new_lbl_idx, orig_act_idx in enumerate(KEEP_ACTIONS):
            for actor in range(ACTORS):
                idx = (chunk, orig_act_idx, actor)
                if sensor_raw[idx].size:
                    data_list.append(sensor_raw[idx].transpose())
                    labels_list.append(new_lbl_idx)
                    batch_ids_list.append(int(rand_perm_idx[idx]))
                    actor_ids_list.append(actor) # NEW

    return (np.array(data_list, dtype=object), np.array(labels_list, dtype=np.int64),
            np.array(batch_ids_list, dtype=np.int64), np.array(actor_ids_list, dtype=np.int64))

def load_sensor_data(sensor="US40", raw_data_path=None):
    raw_data = sio.loadmat(resolve_data_path(raw_data_path))
    if sensor == "all":
        u25, lbl, bat, act = _load_one_sensor_from_raw(raw_data, "US25")
        u33, _, _, _ = _load_one_sensor_from_raw(raw_data, "US33")
        u40, _, _, _ = _load_one_sensor_from_raw(raw_data, "US40")
        stacked = np.array([np.hstack(s) for s in zip(u25, u33, u40)], dtype=object)
        return stacked, lbl, bat, act
    return _load_one_sensor_from_raw(raw_data, sensor=sensor)

def split_5_folds_with_actors(x, y, batch_ids, actor_ids, fold=0):
    test_batches = np.arange(2) + fold * 2
    train_batches = np.delete(np.arange(10), test_batches)
    test_idx = np.where(np.isin(batch_ids, test_batches))[0]
    train_idx = np.where(np.isin(batch_ids, train_batches))[0]
    return x[train_idx], y[train_idx], actor_ids[train_idx], x[test_idx], y[test_idx], actor_ids[test_idx]

def normalize(data):
    all_feats = np.vstack(list(data))
    f_means, f_stds = all_feats.mean(axis=0), all_feats.std(axis=0)
    f_stds[f_stds == 0] = 1.0
    return np.array([(seq - f_means) / f_stds for seq in data], dtype=object)

def pad_for_torch(x, y, max_len=404, mask_val=0.0):
    input_dim, nb_samples = x[0].shape[1], len(x)
    x_pad = np.full((nb_samples, max_len, input_dim), mask_val, dtype=np.float32)
    lengths = np.zeros(nb_samples, dtype=np.int64)
    for i, seq in enumerate(x):
        seq_len = min(len(seq), max_len)
        x_pad[i, :seq_len, :] = seq[:seq_len, :]
        lengths[i] = seq_len
    return x_pad, np.asarray(y, dtype=np.int64), lengths