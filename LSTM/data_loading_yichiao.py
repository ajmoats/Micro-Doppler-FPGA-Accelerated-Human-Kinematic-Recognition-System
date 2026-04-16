"""
Data loading functions for the yichiao test version.
This version does NOT modify the original files.

Main goals:
1. Load data_rot_chunks10.mat robustly
2. Support single-sensor or all-sensor loading
3. Prepare padded tensors for PyTorch LSTM
"""

from pathlib import Path
import numpy as np
import scipy.io as sio

CHUNKS = 10
ACTORS = 13
KEEP_ACTIONS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 16 actions


def resolve_data_path(custom_path=None):
    """
    Resolve the .mat file path.
    Tries:
    1. custom path if provided
    2. repo_root/data_rot_chunks10.mat
    3. repo_root/data/data_rot_chunks10.mat
    """
    if custom_path is not None:
        p = Path(custom_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Custom data path not found: {p}")
        return p

    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        repo_root / "data_rot_chunks10.mat",
        repo_root / "data" / "data_rot_chunks10.mat",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find data_rot_chunks10.mat.\n"
        "Please place it in either:\n"
        f"  {repo_root / 'data_rot_chunks10.mat'}\n"
        f"  {repo_root / 'data' / 'data_rot_chunks10.mat'}"
    )


def _load_one_sensor_from_raw(raw_data, sensor="US40"):
    """
    Load a single sensor from the MATLAB data file.
    Returns:
        data_final: object array of shape (N,), each element is (T, F)
        labels_final: int array of shape (N,)
        batch_ids_final: int array of shape (N,)
    """
    sensor_key = "dataChunks" + sensor
    if sensor_key not in raw_data:
        raise ValueError(f"Sensor {sensor} not found. Valid options: US25, US33, US40")

    sensor_raw = raw_data[sensor_key]
    rand_perm_idx = raw_data["randpermIx"] - 1  # convert MATLAB 1-based to Python 0-based

    data_list = []
    labels_list = []
    batch_ids_list = []

    for chunk in range(CHUNKS):
        for new_label_idx, original_action_idx in enumerate(KEEP_ACTIONS):
            for actor in range(ACTORS):
                idx = (chunk, original_action_idx, actor)

                if sensor_raw[idx].size:
                    # MATLAB -> Python: transpose to (time, features)
                    seq = sensor_raw[idx].transpose()
                    data_list.append(seq)
                    labels_list.append(new_label_idx)
                    batch_ids_list.append(int(rand_perm_idx[idx]))

    data_final = np.array(data_list, dtype=object)
    labels_final = np.array(labels_list, dtype=np.int64)
    batch_ids_final = np.array(batch_ids_list, dtype=np.int64)

    return data_final, labels_final, batch_ids_final


def concatenate(*sensor_arrays):
    """
    Concatenate features from multiple sensors for each sample.
    Each sensor array is expected to be object array of sequences [(T, F), ...]
    """
    return np.array(
        [np.hstack(sample_group) for sample_group in zip(*sensor_arrays)],
        dtype=object,
    )


def load_sensor_data(sensor="US40", raw_data_path=None):
    """
    Load data.
    sensor:
        - 'US25'
        - 'US33'
        - 'US40'
        - 'all'  -> concatenate US25 + US33 + US40
    """
    data_path = resolve_data_path(raw_data_path)
    raw_data = sio.loadmat(data_path)

    if sensor == "all":
        us25, labels25, batch25 = _load_one_sensor_from_raw(raw_data, "US25")
        us33, labels33, batch33 = _load_one_sensor_from_raw(raw_data, "US33")
        us40, labels40, batch40 = _load_one_sensor_from_raw(raw_data, "US40")

        if not (np.array_equal(labels25, labels33) and np.array_equal(labels25, labels40)):
            raise ValueError("Label mismatch across sensors.")
        if not (np.array_equal(batch25, batch33) and np.array_equal(batch25, batch40)):
            raise ValueError("Batch ID mismatch across sensors.")

        stacked = concatenate(us25, us33, us40)
        return stacked, labels25, batch25

    return _load_one_sensor_from_raw(raw_data, sensor=sensor)


def normalize(data):
    """
    Normalize all features to zero mean and unit std across the whole dataset.
    Keeps behavior simple for baseline reproduction.
    """
    all_data = np.vstack(list(data))
    f_means = all_data.mean(axis=0)
    f_stds = all_data.std(axis=0)
    f_stds[f_stds == 0] = 1.0

    normed_data = np.array([(seq - f_means) / f_stds for seq in data], dtype=object)
    return normed_data


def split_5_folds(x, y, batch_ids, fold=0):
    """
    Split into train/valid using the original 5-fold logic.
    """
    nchunks = 10
    nbatches = 5
    ntest = 2

    if fold < 0 or fold >= nbatches:
        raise ValueError("fold must be one of 0,1,2,3,4")

    test_batches = np.arange(ntest) + fold * ntest
    train_batches = np.delete(np.arange(nchunks), test_batches)

    test_idx = np.where(np.isin(batch_ids, test_batches))[0]
    train_idx = np.where(np.isin(batch_ids, train_batches))[0]

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def pad_for_torch(x, y, max_len=404, mask_val=0.0):
    """
    Convert variable-length sequences to fixed-size padded tensors for PyTorch.

    Returns:
        x_pad: (N, max_len, input_dim) float32
        y_out: (N,) int64
        lengths: (N,) int64
    """
    if len(x) == 0:
        raise ValueError("Empty dataset after split.")

    input_dim = x[0].shape[1]
    nb_samples = len(x)

    x_pad = np.full((nb_samples, max_len, input_dim), mask_val, dtype=np.float32)
    lengths = np.zeros(nb_samples, dtype=np.int64)

    for i, seq in enumerate(x):
        seq_len = min(len(seq), max_len)
        x_pad[i, :seq_len, :] = seq[:seq_len, :]
        lengths[i] = seq_len

    y_out = np.asarray(y, dtype=np.int64)
    return x_pad, y_out, lengths


def print_dataset_summary(x, y, name="dataset"):
    """
    Small helper for debugging.
    """
    lengths = np.array([len(seq) for seq in x])
    unique, counts = np.unique(y, return_counts=True)

    print(f"\n[{name}]")
    print(f"num_samples: {len(x)}")
    print(f"num_classes: {len(unique)}")
    print(f"seq_len min/mean/max: {lengths.min()} / {lengths.mean():.2f} / {lengths.max()}")
    print("class counts:")
    for cls, cnt in zip(unique, counts):
        print(f"  class {cls:2d}: {cnt}")