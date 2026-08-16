import os
import json
import numpy as np

from scipy.signal import resample, butter, sosfiltfilt


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS to your actual Bonn dataset path
BONN_DIR = r"C:\Users\HP\Desktop\Bonn\Bonn"

OUTPUT_DIR = "bonn_processed"

ORIGINAL_FS = 173.61
TARGET_FS = 256

WINDOW_SEC = 4
WINDOW_SAMPLES = TARGET_FS * WINDOW_SEC   # 1024


# Bonn classes
SEIZURE_CLASSES = {"S"}
NORMAL_CLASSES = {"F", "N", "O", "Z"}


# ============================================================
# LOAD EEG
# ============================================================

def load_eeg(filepath):

    signal = np.loadtxt(filepath, dtype=np.float64)

    # Ensure 1-D
    signal = signal.reshape(-1)

    return signal


# ============================================================
# RESAMPLE
# ============================================================

def resample_eeg(signal):

    target_length = round(
        len(signal) * TARGET_FS / ORIGINAL_FS
    )

    return resample(
        signal,
        target_length
    )


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal):

    sos = butter(
        5,
        [0.5, 40.0],
        btype="bandpass",
        fs=TARGET_FS,
        output="sos"
    )

    return sosfiltfilt(sos, signal)


# ============================================================
# Z-SCORE NORMALIZATION
# ============================================================

def normalize(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    if std == 0:
        return signal

    return (signal - mean) / std


# ============================================================
# WINDOWING
# ============================================================

def create_windows(signal):

    windows = []

    n_windows = len(signal) // WINDOW_SAMPLES

    for i in range(n_windows):

        start = i * WINDOW_SAMPLES
        end = start + WINDOW_SAMPLES

        window = signal[start:end]

        windows.append(window)

    return np.asarray(windows, dtype=np.float32)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(filepath, label):

    signal = load_eeg(filepath)

    original_samples = len(signal)

    # 1. Resample
    signal = resample_eeg(signal)

    # 2. Bandpass filter
    signal = bandpass_filter(signal)

    # 3. Normalize
    signal = normalize(signal)

    # 4. Create 4-second windows
    windows = create_windows(signal)

    labels = np.full(
        len(windows),
        label,
        dtype=np.int8
    )

    return windows, labels, original_samples


# ============================================================
# MAIN BONN PROCESSING
# ============================================================

def process_bonn():

    all_windows = []
    all_labels = []
    metadata = []

    total_files = 0

    class_summary = {}

    for class_name in sorted(
        SEIZURE_CLASSES | NORMAL_CLASSES
    ):

        folder = os.path.join(
            BONN_DIR,
            class_name
        )

        if not os.path.isdir(folder):

            print(
                f"WARNING: Folder not found: {folder}"
            )

            continue

        label = (
            1
            if class_name in SEIZURE_CLASSES
            else 0
        )

        files = sorted(
            f
            for f in os.listdir(folder)
            if f.lower().endswith(".txt")
        )

        class_windows = 0

        print("\n" + "=" * 60)
        print(
            f"CLASS: {class_name} | LABEL: {label}"
        )
        print(
            f"Files found: {len(files)}"
        )
        print("=" * 60)

        for file_idx, filename in enumerate(files, 1):

            filepath = os.path.join(
                folder,
                filename
            )

            try:

                windows, labels, original_samples = process_file(
                    filepath,
                    label
                )

                all_windows.append(windows)
                all_labels.append(labels)

                class_windows += len(windows)
                total_files += 1

                for _ in range(len(windows)):

                    metadata.append({
                        "dataset": "Bonn",
                        "class": class_name,
                        "file": filename,
                        "label": int(label),
                        "original_samples": original_samples,
                        "original_fs": ORIGINAL_FS,
                        "target_fs": TARGET_FS,
                        "window_samples": WINDOW_SAMPLES,
                        "window_sec": WINDOW_SEC
                    })

                print(
                    f"[{file_idx}/{len(files)}] "
                    f"{filename}: "
                    f"{len(windows)} windows"
                )

            except Exception as e:

                print(
                    f"ERROR: {filename} -> {e}"
                )

        class_summary[class_name] = {
            "label": label,
            "files": len(files),
            "windows": class_windows
        }

    # ========================================================
    # COMBINE
    # ========================================================

    if not all_windows:

        print("\nERROR: No EEG data was processed.")
        return

    X = np.concatenate(
        all_windows,
        axis=0
    )

    y = np.concatenate(
        all_labels,
        axis=0
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "bonn_windows.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "bonn_labels.npy"
        ),
        y
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "summary.json"
        ),
        "w"
    ) as f:

        json.dump(
            class_summary,
            f,
            indent=2
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("BONN PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Files processed      : {total_files}")
    print(f"Total windows        : {len(X)}")
    print(f"Seizure windows      : {np.sum(y == 1)}")
    print(f"Normal windows       : {np.sum(y == 0)}")
    print(f"Window shape         : {X.shape}")
    print(f"Sampling rate        : {TARGET_FS} Hz")
    print(f"Window duration      : {WINDOW_SEC} sec")
    print(f"Samples per window   : {WINDOW_SAMPLES}")

    print("\nClass summary:")

    for cls, info in class_summary.items():

        print(
            f"  {cls}: "
            f"{info['files']} files, "
            f"{info['windows']} windows, "
            f"label={info['label']}"
        )

    print("\nSaved to:")
    print(
        os.path.abspath(OUTPUT_DIR)
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    process_bonn()