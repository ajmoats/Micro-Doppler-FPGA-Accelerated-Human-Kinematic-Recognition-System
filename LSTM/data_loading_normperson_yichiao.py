"""
Person-ID data loading for YiChiao version.

Goal:
- Read all data_rot_*_1.mat files from data/
- Use actions 1~21 only
- Label = person ID
- Support vectorized normalization for speed
- Source-separated stratified folds to prevent data leakage
"""

from pathlib import Path
import re
import numpy as np
import scipy.io as sio

VALID_SENSORS = {"US25", "US33", "US40", "all"}
DEFAULT_ACTION_INDICES = list(range(21)) 

def resolve_data_dir(custom_data_dir=None):
    if custom_data_dir is not None:
        data_dir = Path(custom_data_dir).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return data_dir

def discover_person_files(data_dir, version=1):
    pattern = f"data_rot_*_{version}.mat"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {pattern}")
    return files

def extract_person_initials(file_path):
    m = re.match(r"data_rot_([A-Za-z]+)_\d+\.mat$", file_path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {file_path.name}")
    return m.group(1).lower()

def _safe_label_to_str(x):
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    return str(x)

def _ensure_time_feature(arr, sensor_name):
    expected_feat_dim = 327 if sensor_name == "US25" else 328
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.size == 0: return None
    if arr.ndim == 1: arr = arr[:, None]
    
    if arr.shape[0] == expected_feat_dim and arr.shape[1] != expected_feat_dim:
        arr = arr.T 
    return arr.astype(np.float32)

def _get_sensor_action_matrix(mat_dict, sensor_name, action_idx):
    key_map = {"US25": "us25_data", "US33": "us33_data", "US40": "us40_data"}
    key = key_map[sensor_name]
    raw_arr = mat_dict[key][action_idx, 0]
    return _ensure_time_feature(raw_arr, sensor_name)

def _build_one_sample(mat_dict, sensor, action_idx):
    if sensor == "all":
        us25 = _get_sensor_action_matrix(mat_dict, "US25", action_idx)
        us33 = _get_sensor_action_matrix(mat_dict, "US33", action_idx)
        us40 = _get_sensor_action_matrix(mat_dict, "US40", action_idx)
        if us25 is None or us33 is None or us40 is None: return None
        t_min = min(us25.shape[0], us33.shape[0], us40.shape[0])
        return np.hstack([us25[:t_min, :], us33[:t_min, :], us40[:t_min, :]]).astype(np.float32)
    return _get_sensor_action_matrix(mat_dict, sensor, action_idx)

def load_person_dataset(sensor="US40", version=1, action_indices=None, data_dir=None, window_len=None, stride=None):
    if sensor not in VALID_SENSORS: raise ValueError(f"sensor must be one of {VALID_SENSORS}")
    action_indices = action_indices or DEFAULT_ACTION_INDICES
    data_dir = resolve_data_dir(data_dir)
    files = discover_person_files(data_dir, version=version)

    unique_persons = sorted(set(extract_person_initials(f) for f in files))
    person_to_id = {p: i for i, p in enumerate(unique_persons)}
    id_to_person = {i: p for p, i in person_to_id.items()}

    data_list, label_list, metadata = [], [], []
    action_names = None

    for file_path in files:
        person = extract_person_initials(file_path)
        person_id = person_to_id[person]
        try:
            mat_dict = sio.loadmat(file_path)
        except Exception: continue # Skip corrupted files

        if action_names is None:
            raw_labels = mat_dict["lblMaster"]
            action_names = [_safe_label_to_str(raw_labels[i, 0]) for i in range(raw_labels.shape[0])]

        for action_idx in action_indices:
            sample = _build_one_sample(mat_dict, sensor, action_idx)
            if sample is None: continue

            if window_len and stride:
                windows = make_windows(sample, window_len, stride)
                for w_idx, w in enumerate(windows):
                    data_list.append(w)
                    label_list.append(person_id)
                    metadata.append({"file_name": file_path.name, "person": person, "person_id": person_id, "action_name": action_names[action_idx], "window_idx": w_idx})
            else:
                data_list.append(sample)
                label_list.append(person_id)
                metadata.append({"file_name": file_path.name, "person": person, "person_id": person_id, "action_name": action_names[action_idx]})

    return np.array(data_list, dtype=object), np.array(label_list, dtype=np.int64), metadata, person_to_id, id_to_person, action_names

def normalize(data, metadata=None):
    """ Vectorized normalization to speed up training significantly. """
    all_data_stacked = np.concatenate(data, axis=0).astype(np.float32)
    feat_mean = all_data_stacked.mean(axis=0)
    feat_std = all_data_stacked.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    return np.array([(seq.astype(np.float32) - feat_mean) / feat_std for seq in data], dtype=object)

def make_stratified_folds(metadata, n_splits=5, seed=1337):
    """
    SOURCE-SEPARATED Stratified Folds.
    Groups windows by their original source file to prevent data leakage.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Map files to their owners (Person IDs)
    file_to_person = {}
    for meta in metadata:
        fname = meta['file_name']
        if fname not in file_to_person:
            file_to_person[fname] = meta['person_id']

    unique_files = np.array(list(file_to_person.keys()))
    file_labels = np.array([file_to_person[f] for f in unique_files])

    # 2. Stratify the FILES, not the windows
    folds_files = [[] for _ in range(n_splits)]
    for person_id in np.unique(file_labels):
        person_files = unique_files[file_labels == person_id]
        rng.shuffle(person_files)
        
        # Distribute files across folds
        for i, f in enumerate(person_files):
            folds_files[i % n_splits].append(f)

    # 3. Map the chosen files back to window indices
    final_fold_indices = []
    for fold_id in range(n_splits):
        target_files = set(folds_files[fold_id])
        idx_in_fold = [i for i, m in enumerate(metadata) if m['file_name'] in target_files]
        final_fold_indices.append(np.array(idx_in_fold, dtype=np.int64))

    return final_fold_indices

def get_fold_split(x, y, metadata, fold_indices, fold=0):
    test_idx = fold_indices[fold]
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx], [metadata[i] for i in train_idx], [metadata[i] for i in test_idx]

def pad_for_torch(x, y, max_len=None, mask_val=0.0):
    if max_len is None: max_len = max(s.shape[0] for s in x)
    x_pad = np.full((len(x), max_len, x[0].shape[1]), mask_val, dtype=np.float32)
    lengths = np.zeros(len(x), dtype=np.int64)
    for i, seq in enumerate(x):
        l = min(seq.shape[0], max_len)
        x_pad[i, :l, :] = seq[:l, :]
        lengths[i] = l
    return x_pad, np.asarray(y, dtype=np.int64), lengths

def make_windows(seq, window_len=200, stride=100):
    T, F = seq.shape
    if T < window_len: return []
    return [seq[s:s+window_len, :] for s in range(0, T - window_len + 1, stride)]