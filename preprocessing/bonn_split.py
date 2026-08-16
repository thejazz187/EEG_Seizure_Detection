import os
import json
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "bonn_processed"
OUTPUT_DIR = "bonn_splits"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

X = np.load(
    os.path.join(
        INPUT_DIR,
        "bonn_windows.npy"
    )
)

y = np.load(
    os.path.join(
        INPUT_DIR,
        "bonn_labels.npy"
    )
)

with open(
    os.path.join(
        INPUT_DIR,
        "metadata.json"
    ),
    "r"
) as f:

    metadata = json.load(f)


print("=" * 60)
print("LOADED BONN DATA")
print("=" * 60)

print("Windows :", X.shape)
print("Labels  :", y.shape)


# ============================================================
# GROUP WINDOWS BY ORIGINAL RECORDING
# ============================================================

recordings = {}

for index, item in enumerate(metadata):

    filename = item["file"]

    if filename not in recordings:

        recordings[filename] = {
            "indices": [],
            "label": item["label"],
            "class": item["class"]
        }

    recordings[filename]["indices"].append(index)


print("\nTotal recordings:", len(recordings))


# ============================================================
# GET RECORDING-LEVEL INFORMATION
# ============================================================

recording_names = list(recordings.keys())

recording_labels = np.array([
    recordings[name]["label"]
    for name in recording_names
])


# ============================================================
# TRAIN / TEMP SPLIT
# ============================================================

train_names, temp_names = train_test_split(
    recording_names,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=recording_labels,
    random_state=RANDOM_STATE
)


# ============================================================
# VALIDATION / TEST SPLIT
# ============================================================

temp_labels = np.array([
    recordings[name]["label"]
    for name in temp_names
])


relative_test_ratio = (
    TEST_RATIO /
    (VAL_RATIO + TEST_RATIO)
)


val_names, test_names = train_test_split(
    temp_names,
    test_size=relative_test_ratio,
    stratify=temp_labels,
    random_state=RANDOM_STATE
)


# ============================================================
# CONVERT RECORDING NAMES → WINDOW INDICES
# ============================================================

def get_window_indices(names):

    indices = []

    for name in names:

        indices.extend(
            recordings[name]["indices"]
        )

    return np.array(
        sorted(indices),
        dtype=np.int64
    )


train_idx = get_window_indices(train_names)
val_idx = get_window_indices(val_names)
test_idx = get_window_indices(test_names)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE DATA
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X[train_idx]
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y[train_idx]
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_val.npy"
    ),
    X[val_idx]
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_val.npy"
    ),
    y[val_idx]
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X[test_idx]
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y[test_idx]
)


# ============================================================
# SAVE SPLIT INFORMATION
# ============================================================

split_info = {

    "train_recordings": train_names,
    "validation_recordings": val_names,
    "test_recordings": test_names,

    "train_windows": int(len(train_idx)),
    "validation_windows": int(len(val_idx)),
    "test_windows": int(len(test_idx)),

    "random_state": RANDOM_STATE
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "split_info.json"
    ),
    "w"
) as f:

    json.dump(
        split_info,
        f,
        indent=2
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_distribution(name, labels):

    seizure = np.sum(labels == 1)
    normal = np.sum(labels == 0)

    print(f"\n{name}")
    print("-" * 40)
    print("Windows :", len(labels))
    print("Normal  :", normal)
    print("Seizure :", seizure)


print("\n")
print("=" * 60)
print("RECORDING-LEVEL SPLIT COMPLETE")
print("=" * 60)

print(
    f"Training recordings   : {len(train_names)}"
)

print(
    f"Validation recordings : {len(val_names)}"
)

print(
    f"Testing recordings    : {len(test_names)}"
)


print_distribution(
    "TRAIN",
    y[train_idx]
)

print_distribution(
    "VALIDATION",
    y[val_idx]
)

print_distribution(
    "TEST",
    y[test_idx]
)


print("\nSaved to:")
print(
    os.path.abspath(OUTPUT_DIR)
)

print("=" * 60)