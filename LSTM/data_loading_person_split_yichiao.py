"""
Person-ID data loading for YiChiao version.

Goal:
- Read all data_rot_*_1.mat files from data/
- Use actions 1~21 only (exclude the last 2 freestyle actions)
- Label = person ID, not action ID
- Support sensor = US25 / US33 / US40 / all
"""

from pathlib import Path
import re
import numpy as np
import scipy.io as sio


VALID_SENSORS = {"US25", "US33", "US40", "all"}
DEFAULT_ACTION_INDICES = list(range(21))   # use actions 1~21 only (0-based: 0~20)


def resolve_data_dir(custom_data_dir=None):
    """
    Resolve repo_root/data directory.
    """
    if custom_data_dir is not None:
        data_dir = Path(custom_data_dir).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root / "data"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    return data_dir


def discover_person_files(data_dir, version=1):
    """
    Find files like data_rot_dm_1.mat
    """
    pattern = f"data_rot_*_{version}.mat"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {pattern}")
    return files


def extract_person_initials(file_path):
    """
    Example:
    data_rot_dm_1.mat -> dm
    """
    m = re.match(r"data_rot_([A-Za-z]+)_\d+\.mat$", file_path.name)
    if not m:
        raise ValueError(f"Unexpected filename format: {file_path.name}")
    return m.group(1).lower()


def _safe_label_to_str(x):
    """
    Convert MATLAB-loaded label cell content to Python string.
    """
    while isinstance(x, np.ndarray) and x.size == 1:
        x = x.item()
    return str(x)


def _ensure_time_feature(arr, sensor_name):
    """
    Convert one MATLAB cell entry to a clean (T, F) float32 array.
    Return None if the entry is empty.
    """
    expected_feat_dim = 327 if sensor_name == "US25" else 328

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)

    # empty sample
    if arr.size == 0:
        return None

    if arr.ndim == 1:
        arr = arr[:, None]

    if arr.ndim != 2:
        raise ValueError(
            f"Unexpected array ndim for {sensor_name}: shape={arr.shape}"
        )

    # Case A: original MATLAB-style (F, T)
    if arr.shape[0] == expected_feat_dim and arr.shape[1] != expected_feat_dim:
        arr = arr.T   # -> (T, F)

    # Case B: already (T, F)
    elif arr.shape[1] == expected_feat_dim:
        pass

    # Case C: weird row vector
    elif arr.shape[0] == 1 and arr.shape[1] > 1:
        arr = arr.T

    # Case D: weird column vector
    elif arr.shape[1] == 1 and arr.shape[0] > 1:
        pass

    else:
        raise ValueError(
            f"Cannot determine orientation for {sensor_name}: shape={arr.shape}"
        )

    return arr.astype(np.float32)


def _get_sensor_action_matrix(mat_dict, sensor_name, action_idx):
    key_map = {
        "US25": "us25_data",
        "US33": "us33_data",
        "US40": "us40_data",
    }
    key = key_map[sensor_name]
    raw_arr = mat_dict[key][action_idx, 0]
    return _ensure_time_feature(raw_arr, sensor_name)


def _build_one_sample(mat_dict, sensor, action_idx):
    """
    Returns one sample in shape (T, F)
    Return None if the sample is empty.
    """
    if sensor == "all":
        us25 = _get_sensor_action_matrix(mat_dict, "US25", action_idx)
        us33 = _get_sensor_action_matrix(mat_dict, "US33", action_idx)
        us40 = _get_sensor_action_matrix(mat_dict, "US40", action_idx)

        # if any sensor is empty, skip this sample
        if us25 is None or us33 is None or us40 is None:
            return None

        t_min = min(us25.shape[0], us33.shape[0], us40.shape[0])
        us25 = us25[:t_min, :]
        us33 = us33[:t_min, :]
        us40 = us40[:t_min, :]

        sample = np.hstack([us25, us33, us40]).astype(np.float32)
        return sample

    x = _get_sensor_action_matrix(mat_dict, sensor, action_idx)
    if x is None:
        return None

    return x


def load_person_dataset(
    sensor="US40",
    version=1,
    action_indices=None,
    data_dir=None,
    window_len=None,
    stride=None,
):
    """
    Build person-ID dataset.

    Returns:
        x: object array, each item shape (T, F)
        y: int array, person labels
        metadata: list[dict]
        person_to_id: dict
        id_to_person: dict
        action_names: list[str]
    """
    if sensor not in VALID_SENSORS:
        raise ValueError(f"sensor must be one of {VALID_SENSORS}")

    if action_indices is None:
        action_indices = DEFAULT_ACTION_INDICES

    data_dir = resolve_data_dir(data_dir)
    files = discover_person_files(data_dir, version=version)

    persons = [extract_person_initials(f) for f in files]
    unique_persons = sorted(set(persons))
    person_to_id = {p: i for i, p in enumerate(unique_persons)}
    id_to_person = {i: p for p, i in person_to_id.items()}

    data_list = []
    label_list = []
    metadata = []

    action_names = None

    for file_path in files:
        person = extract_person_initials(file_path)
        person_id = person_to_id[person]

        mat_dict = sio.loadmat(file_path)

        if action_names is None:
            raw_labels = mat_dict["lblMaster"]
            action_names = [_safe_label_to_str(raw_labels[i, 0]) for i in range(raw_labels.shape[0])]

        for action_idx in action_indices:
            sample = _build_one_sample(mat_dict, sensor=sensor, action_idx=action_idx)

            if sample is None:
                print(
                    f"Skipping empty sample | file={file_path.name} | "
                    f"person={person} | action_idx={action_idx+1} | "
                    f"action_name={action_names[action_idx]} | sensor={sensor}"
                )
                continue

            # windowed version
            if window_len is not None and stride is not None:
                windows = make_windows(sample, window_len=window_len, stride=stride)

                if len(windows) == 0:
                    print(
                        f"Skipping too-short sample | file={file_path.name} | "
                        f"person={person} | action_idx={action_idx+1} | "
                        f"action_name={action_names[action_idx]} | sensor={sensor} | "
                        f"seq_len={sample.shape[0]}"
                    )
                    continue

                for w_idx, w in enumerate(windows):
                    data_list.append(w)
                    label_list.append(person_id)
                    metadata.append(
                        {
                            "file_name": file_path.name,
                            "person": person,
                            "person_id": person_id,
                            "action_idx_1based": action_idx + 1,
                            "action_name": action_names[action_idx],
                            "sensor": sensor,
                            "seq_len": int(w.shape[0]),
                            "feat_dim": int(w.shape[1]),
                            "window_idx": w_idx,
                        }
                    )
            else:
                data_list.append(sample)
                label_list.append(person_id)
                metadata.append(
                    {
                        "file_name": file_path.name,
                        "person": person,
                        "person_id": person_id,
                        "action_idx_1based": action_idx + 1,
                        "action_name": action_names[action_idx],
                        "sensor": sensor,
                        "seq_len": int(sample.shape[0]),
                        "feat_dim": int(sample.shape[1]),
                    }
                )

    x = np.array(data_list, dtype=object)
    y = np.array(label_list, dtype=np.int64)

    return x, y, metadata, person_to_id, id_to_person, action_names


def normalize(data, metadata=None):
    """
    Normalize using global feature mean/std.
    data is object array of (T, F)
    """
    seqs = []
    feat_dims = []

    for i, seq in enumerate(data):
        seq = np.asarray(seq, dtype=np.float32)

        if seq.ndim != 2:
            msg = f"Sample {i} is not 2D: shape={seq.shape}"
            if metadata is not None:
                msg += (
                    f" | file={metadata[i]['file_name']}"
                    f" | person={metadata[i]['person']}"
                    f" | action={metadata[i]['action_name']}"
                )
            raise ValueError(msg)

        seqs.append(seq)
        feat_dims.append(seq.shape[1])

    unique_feat_dims = sorted(set(feat_dims))
    if len(unique_feat_dims) != 1:
        print("Feature dimension mismatch found:")
        for i, seq in enumerate(seqs):
            if metadata is not None:
                print(
                    f"[{i}] shape={seq.shape} | "
                    f"file={metadata[i]['file_name']} | "
                    f"person={metadata[i]['person']} | "
                    f"action={metadata[i]['action_name']}"
                )
            else:
                print(f"[{i}] shape={seq.shape}")
        raise ValueError(f"Inconsistent feature dims: {unique_feat_dims}")

    all_data = np.concatenate(seqs, axis=0)
    feat_mean = all_data.mean(axis=0)
    feat_std = all_data.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    normed = np.array([(seq - feat_mean) / feat_std for seq in seqs], dtype=object)
    return normed


def make_stratified_folds(y, n_splits=5, seed=1337):
    """
    Manual stratified folds without sklearn.
    Returns a list of test-index arrays.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    folds = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)

        parts = np.array_split(cls_idx, n_splits)
        for fold_id in range(n_splits):
            folds[fold_id].extend(parts[fold_id].tolist())

    folds = [np.array(sorted(f), dtype=np.int64) for f in folds]
    return folds


def get_fold_split(x, y, metadata, fold_indices, fold=0):
    """
    fold_indices = output of make_stratified_folds(...)
    """
    test_idx = fold_indices[fold]
    all_idx = np.arange(len(y))
    train_idx = np.setdiff1d(all_idx, test_idx)

    train_x = x[train_idx]
    train_y = y[train_idx]
    valid_x = x[test_idx]
    valid_y = y[test_idx]

    train_meta = [metadata[i] for i in train_idx]
    valid_meta = [metadata[i] for i in test_idx]

    return train_x, train_y, valid_x, valid_y, train_meta, valid_meta


def pad_for_torch(x, y, max_len=None, mask_val=0.0):
    """
    Convert object-array sequences to padded arrays for PyTorch.

    If max_len is None, use the max sequence length in x.
    """
    if len(x) == 0:
        raise ValueError("Empty dataset")

    if max_len is None:
        max_len = max(seq.shape[0] for seq in x)

    input_dim = x[0].shape[1]
    n = len(x)

    x_pad = np.full((n, max_len, input_dim), mask_val, dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)

    for i, seq in enumerate(x):
        seq_len = min(seq.shape[0], max_len)
        x_pad[i, :seq_len, :] = seq[:seq_len, :]
        lengths[i] = seq_len

    y_out = np.asarray(y, dtype=np.int64)
    return x_pad, y_out, lengths


def print_dataset_summary(x, y, metadata, id_to_person, title="dataset"):
    """
    Print a useful summary for debugging.
    """
    lengths = np.array([seq.shape[0] for seq in x])
    feat_dims = np.array([seq.shape[1] for seq in x])

    print(f"\n[{title}]")
    print(f"num_samples: {len(x)}")
    print(f"num_classes: {len(np.unique(y))}")
    print(f"seq_len min/mean/max: {lengths.min()} / {lengths.mean():.2f} / {lengths.max()}")
    print(f"feat_dim unique: {sorted(set(feat_dims.tolist()))}")

    unique_y, counts = np.unique(y, return_counts=True)
    print("samples per person:")
    for cls, cnt in zip(unique_y, counts):
        print(f"  {id_to_person[int(cls)]}: {cnt}")

    action_count = {}
    for m in metadata:
        action_count[m["action_name"]] = action_count.get(m["action_name"], 0) + 1

    print("samples per action:")
    for action_name, cnt in action_count.items():
        print(f"  {action_name}: {cnt}")


def make_windows(seq, window_len=200, stride=100):
    """
    seq: (T, F)
    returns a list of windows, each shape (window_len, F)
    """
    T, F = seq.shape

    if T < window_len:
        return []

    windows = []
    for start in range(0, T - window_len + 1, stride):
        end = start + window_len
        windows.append(seq[start:end, :])

    return windows

def get_file_based_split(x, y, metadata, test_person_files):
    """
    Priority 4: Cleaner Source-Separated Evaluation.
    Ensures windows from the same recording are not shared between train/valid.
    """
    train_idx = []
    test_idx = []

    for i, meta in enumerate(metadata):
        if meta["file_name"] in test_person_files:
            test_idx.append(i)
        else:
            train_idx.append(i)

    return x[np.array(train_idx)], y[np.array(train_idx)], \
           x[np.array(test_idx)], y[np.array(test_idx)], \
           [metadata[i] for i in train_idx], [metadata[i] for i in test_idx]


def get_cross_session_split(x, y, metadata, train_version=1, test_version=2):
    """
    Priority 3: Cross-Session Testing.
    Train on all Version 1 files and test on all Version 2 files.
    """
    train_idx = []
    test_idx = []

    for i, meta in enumerate(metadata):
        # Extracts version number from filename (e.g., data_rot_dm_1.mat -> 1)
        version_match = re.search(r'_(\d+)\.mat$', meta["file_name"])
        if version_match:
            version = int(version_match.group(1))
            if version == train_version:
                train_idx.append(i)
            elif version == test_version:
                test_idx.append(i)

    return x[np.array(train_idx)], y[np.array(train_idx)], \
           x[np.array(test_idx)], y[np.array(test_idx)], \
           [metadata[i] for i in train_idx], [metadata[i] for i in test_idx]