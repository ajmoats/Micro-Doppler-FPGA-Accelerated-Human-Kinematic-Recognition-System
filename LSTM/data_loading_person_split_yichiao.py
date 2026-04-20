"""
Person-ID data loading for YiChiao version.
Supports version-based splitting for Cross-Session testing.
"""
from pathlib import Path
import re
import numpy as np
import scipy.io as sio

VALID_SENSORS = {"US25", "US33", "US40", "all"}
DEFAULT_ACTION_INDICES = list(range(21))

def resolve_data_dir(custom_data_dir=None):
    if custom_data_dir: return Path(custom_data_dir).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data"

def discover_person_files(data_dir, version=1):
    files = sorted(data_dir.glob(f"data_rot_*_{version}.mat"))
    if not files: raise FileNotFoundError(f"No V{version} files found in {data_dir}")
    return files

def extract_person_initials(file_path):
    return re.match(r"data_rot_([A-Za-z]+)_\d+\.mat$", file_path.name).group(1).lower()

def _ensure_time_feature(arr, sensor_name):
    expected = 327 if sensor_name == "US25" else 328
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.size == 0: return None
    if arr.ndim == 1: arr = arr[:, None]
    if arr.shape[0] == expected: arr = arr.T
    return arr

def _build_one_sample(mat_dict, sensor, action_idx):
    if sensor == "all":
        senses = [(_ensure_time_feature(mat_dict[k][action_idx, 0], n)) 
                  for k, n in [("us25_data","US25"), ("us33_data","US33"), ("us40_data","US40")]]
        if any(s is None for s in senses): return None
        t_min = min(s.shape[0] for s in senses)
        return np.hstack([s[:t_min, :] for s in senses]).astype(np.float32)
    return _ensure_time_feature(mat_dict[f"{sensor.lower()}_data"][action_idx, 0], sensor)

def load_person_dataset(sensor="all", version=1, action_indices=None, data_dir=None, person_to_id=None):
    data_dir = resolve_data_dir(data_dir)
    files = discover_person_files(data_dir, version)
    
    if person_to_id is None:
        persons = sorted(set(extract_person_initials(f) for f in files))
        person_to_id = {p: i for i, p in enumerate(persons)}
    
    data, labels, meta = [], [], []
    for f in files:
        p = extract_person_initials(f)
        if p not in person_to_id: continue
        mat = sio.loadmat(f)
        for idx in (action_indices or DEFAULT_ACTION_INDICES):
            s = _build_one_sample(mat, sensor, idx)
            if s is not None:
                data.append(s)
                labels.append(person_to_id[p])
                meta.append({"file_name": f.name, "person": p})
                
    return np.array(data, dtype=object), np.array(labels, dtype=np.int64), meta, person_to_id, {i:p for p,i in person_to_id.items()}, None

def normalize(data, meta=None):
    flat = np.concatenate(data, axis=0)
    mu, sigma = flat.mean(0), flat.std(0)
    sigma[sigma == 0] = 1.0
    return np.array([(s - mu) / sigma for s in data], dtype=object)

def pad_for_torch(x, y, max_len=None):
    if max_len is None: max_len = max(s.shape[0] for s in x)
    out = np.zeros((len(x), max_len, x[0].shape[1]), dtype=np.float32)
    lens = np.zeros(len(x), dtype=np.int64)
    for i, s in enumerate(x):
        l = min(s.shape[0], max_len)
        out[i, :l, :] = s[:l, :]
        lens[i] = l
    return out, np.array(y, dtype=np.int64), lens

def get_cross_session_split(x, y, meta, train_version=1, test_version=2):
    tr_idx = [i for i, m in enumerate(meta) if f"_{train_version}.mat" in m["file_name"]]
    te_idx = [i for i, m in enumerate(meta) if f"_{test_version}.mat" in m["file_name"]]
    return x[tr_idx], y[tr_idx], x[te_idx], y[te_idx], [meta[i] for i in tr_idx], [meta[i] for i in te_idx]

def print_dataset_summary(x, y, meta, id2p, title="Set"):
    print(f"\n[{title}] Samples: {len(x)} | People: {len(np.unique(y))}")