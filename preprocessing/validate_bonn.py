import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = "bonn_processed"
SPLIT_DIR = "bonn_splits"


# ============================================================
# 1. LOAD AND CHECK SHAPES / NaN / INF
# ============================================================

print("=" * 60)
print("1. DATA INTEGRITY CHECK")
print("=" * 60)

for split in ["train", "val", "test"]:

    X = np.load(
        os.path.join(
            SPLIT_DIR,
            f"X_{split}.npy"
        )
    )

    y = np.load(
        os.path.join(
            SPLIT_DIR,
            f"y_{split}.npy"
        )
    )

    print(f"\n{split.upper()}")

    print("X shape :", X.shape)
    print("y shape :", y.shape)

    print("NaN    :", np.isnan(X).sum())
    print("Inf    :", np.isinf(X).sum())

    print("Min    :", X.min())
    print("Max    :", X.max())
    print("Mean   :", X.mean())
    print("Std    :", X.std())

    print(
        "Classes:",
        np.unique(y, return_counts=True)
    )


# ============================================================
# 2. RECORDING-LEVEL SPLIT CHECK
# ============================================================

print("\n" + "=" * 60)
print("2. RECORDING-LEVEL LEAKAGE CHECK")
print("=" * 60)

with open(
    os.path.join(
        SPLIT_DIR,
        "split_info.json"
    ),
    "r"
) as f:

    info = json.load(f)


train = set(
    info["train_recordings"]
)

val = set(
    info["validation_recordings"]
)

test = set(
    info["test_recordings"]
)


train_val = train & val
train_test = train & test
val_test = val & test


print(
    "Train ∩ Validation:",
    train_val
)

print(
    "Train ∩ Test:",
    train_test
)

print(
    "Validation ∩ Test:",
    val_test
)


if not train_val and not train_test and not val_test:

    print(
        "\nPASS: No recording appears in multiple splits."
    )

else:

    print(
        "\nFAIL: Recording leakage detected!"
    )


# ============================================================
# 3. DUPLICATE WINDOW CHECK
# ============================================================

print("\n" + "=" * 60)
print("3. DUPLICATE WINDOW CHECK")
print("=" * 60)

X_all = np.load(
    os.path.join(
        PROCESSED_DIR,
        "bonn_windows.npy"
    )
)

unique_windows = len(
    np.unique(
        X_all,
        axis=0
    )
)

total_windows = len(X_all)

duplicates = (
    total_windows -
    unique_windows
)


print(
    "Total windows  :",
    total_windows
)

print(
    "Unique windows  :",
    unique_windows
)

print(
    "Duplicate windows:",
    duplicates
)


# ============================================================
# 4. PLOT NORMAL EEG
# ============================================================

print("\n" + "=" * 60)
print("4. EEG VISUAL CHECK")
print("=" * 60)

X_train = np.load(
    os.path.join(
        SPLIT_DIR,
        "X_train.npy"
    )
)

y_train = np.load(
    os.path.join(
        SPLIT_DIR,
        "y_train.npy"
    )
)


normal_indices = np.where(
    y_train == 0
)[0]

seizure_indices = np.where(
    y_train == 1
)[0]


normal_idx = normal_indices[0]
seizure_idx = seizure_indices[0]


plt.figure(figsize=(12, 4))

plt.plot(
    X_train[normal_idx]
)

plt.title(
    "Bonn - Normal EEG Window"
)

plt.xlabel(
    "Sample (256 Hz)"
)

plt.ylabel(
    "Normalized Amplitude"
)

plt.tight_layout()

plt.savefig(
    "bonn_normal_example.png",
    dpi=200
)

plt.show()


# ============================================================
# 5. PLOT SEIZURE EEG
# ============================================================

plt.figure(figsize=(12, 4))

plt.plot(
    X_train[seizure_idx]
)

plt.title(
    "Bonn - Seizure EEG Window"
)

plt.xlabel(
    "Sample (256 Hz)"
)

plt.ylabel(
    "Normalized Amplitude"
)

plt.tight_layout()

plt.savefig(
    "bonn_seizure_example.png",
    dpi=200
)

plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BONN VALIDATION COMPLETE")
print("=" * 60)

if not train_val and not train_test and not val_test:

    print("✓ Recording-level split: PASS")

else:

    print("✗ Recording-level split: FAIL")


print(
    f"✓ Total windows checked: {total_windows}"
)

print(
    f"✓ Duplicate windows: {duplicates}"
)

print(
    "\nCheck the two generated plots manually."
)

print("=" * 60)