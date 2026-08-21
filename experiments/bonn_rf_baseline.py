"""
Bonn EEG Random Forest Baseline

Standardized RF configuration:

    n_estimators     = 300
    max_depth        = 12
    min_samples_leaf = 5
    class_weight     = balanced
    random_state     = 42
    n_jobs            = -1

Feature representation:

    23 features/channel
    mean + std + max
    = 69 features

Threshold:

    Youden's J on validation set

Final evaluation:

    Test set using frozen validation threshold
"""

import os
import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# Import shared feature extractor
from models.eeg_features import (
    extract_features_batch,
    feature_names,
)


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Change this ONLY if your Bonn split folder is elsewhere.

SPLIT_DIR = (
    r"C:\Users\HP\Desktop\Bonn\Bonn\bonn_splits"
)

FS = 256

RANDOM_STATE = 42


# Same RF configuration across datasets
RF_CONFIG = {

    "n_estimators": 300,

    "max_depth": 12,

    "min_samples_leaf": 5,

    "class_weight": "balanced",

    "random_state": 42,

    "n_jobs": -1,
}


# ============================================================
# LOAD BONN SPLITS
# ============================================================

def load_bonn():

    print("=" * 70)
    print("LOADING BONN DATA")
    print("=" * 70)

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

    X_val = np.load(
        os.path.join(
            SPLIT_DIR,
            "X_val.npy"
        )
    )

    y_val = np.load(
        os.path.join(
            SPLIT_DIR,
            "y_val.npy"
        )
    )

    X_test = np.load(
        os.path.join(
            SPLIT_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            SPLIT_DIR,
            "y_test.npy"
        )
    )

    print("\nTRAIN")
    print("X:", X_train.shape)
    print("y:", y_train.shape)

    print("\nVALIDATION")
    print("X:", X_val.shape)
    print("y:", y_val.shape)

    print("\nTEST")
    print("X:", X_test.shape)
    print("y:", y_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

def create_features(
    X_train,
    X_val,
    X_test
):

    print("\n" + "=" * 70)
    print("EXTRACTING STANDARDIZED EEG FEATURES")
    print("=" * 70)

    print("\nExtracting TRAIN features...")

    X_train_features = (
        extract_features_batch(
            X_train,
            fs=FS
        )
    )

    print("Extracting VALIDATION features...")

    X_val_features = (
        extract_features_batch(
            X_val,
            fs=FS
        )
    )

    print("Extracting TEST features...")

    X_test_features = (
        extract_features_batch(
            X_test,
            fs=FS
        )
    )

    print("\nFeature shapes:")

    print(
        "Train:",
        X_train_features.shape
    )

    print(
        "Validation:",
        X_val_features.shape
    )

    print(
        "Test:",
        X_test_features.shape
    )

    # --------------------------------------------------------
    # VERY IMPORTANT CHECK
    # --------------------------------------------------------

    assert (
        X_train_features.shape[1]
        == 69
    )

    assert (
        X_val_features.shape[1]
        == 69
    )

    assert (
        X_test_features.shape[1]
        == 69
    )

    print(
        "\nPASS: All Bonn windows have 69 features."
    )

    return (
        X_train_features,
        X_val_features,
        X_test_features,
    )


# ============================================================
# YOUDEN'S J
# ============================================================

def find_youden_threshold(
    y_true,
    probabilities
):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities
    )

    youden_j = (
        tpr - fpr
    )

    best_index = np.argmax(
        youden_j
    )

    threshold = thresholds[
        best_index
    ]

    return threshold


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    y_true,
    probabilities,
    threshold,
    name
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auroc = roc_auc_score(
        y_true,
        probabilities
    )

    auprc = average_precision_score(
        y_true,
        probabilities
    )

    print("\n" + "=" * 70)
    print(f"{name.upper()} RESULTS")
    print("=" * 70)

    print(
        f"Threshold    : {threshold:.6f}"
    )

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Precision    : {precision:.4f}"
    )

    print(
        f"Sensitivity  : {sensitivity:.4f}"
    )

    print(
        f"Specificity  : {specificity:.4f}"
    )

    print(
        f"F1 Score     : {f1:.4f}"
    )

    print(
        f"AUROC        : {auroc:.4f}"
    )

    print(
        f"AUPRC        : {auprc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "threshold": threshold,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1
    # Load preprocessed Bonn data
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_bonn()


    # --------------------------------------------------------
    # STEP 2
    # Convert EEG windows → 69 features
    # --------------------------------------------------------

    (
        X_train_features,
        X_val_features,
        X_test_features,
    ) = create_features(
        X_train,
        X_val,
        X_test,
    )


    # --------------------------------------------------------
    # STEP 3
    # Train Random Forest
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    print("\nConfiguration:")

    for key, value in RF_CONFIG.items():

        print(
            f"{key:20s}: {value}"
        )

    rf = RandomForestClassifier(
        **RF_CONFIG
    )

    rf.fit(
        X_train_features,
        y_train
    )

    print(
        "\nPASS: Random Forest training complete."
    )


    # --------------------------------------------------------
    # STEP 4
    # Validation probabilities
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD SELECTION")
    print("=" * 70)

    val_probabilities = (
        rf.predict_proba(
            X_val_features
        )[:, 1]
    )


    # --------------------------------------------------------
    # STEP 5
    # Youden's J
    # --------------------------------------------------------

    threshold = (
        find_youden_threshold(
            y_val,
            val_probabilities
        )
    )

    print(
        f"\nYouden's J threshold = "
        f"{threshold:.6f}"
    )


    # --------------------------------------------------------
    # STEP 6
    # Evaluate validation
    # --------------------------------------------------------

    val_results = evaluate(
        y_val,
        val_probabilities,
        threshold,
        "Validation"
    )


    # --------------------------------------------------------
    # STEP 7
    # FINAL TEST
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_probabilities = (
        rf.predict_proba(
            X_test_features
        )[:, 1]
    )


    # IMPORTANT:
    #
    # We use the threshold learned from VALIDATION.
    #
    # We do NOT tune threshold using TEST.

    test_results = evaluate(
        y_test,
        test_probabilities,
        threshold,
        "Test"
    )


    # --------------------------------------------------------
    # STEP 8
    # Feature importance
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP 15 RANDOM FOREST FEATURES")
    print("=" * 70)

    names = feature_names()

    importances = (
        rf.feature_importances_
    )

    top_indices = np.argsort(
        importances
    )[::-1][:15]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{names[index]:35s}"
            f"{importances[index]:.6f}"
        )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BONN RANDOM FOREST BASELINE COMPLETE")
    print("=" * 70)

    print("\nFINAL TEST RESULTS")

    for key, value in test_results.items():

        print(
            f"{key:15s}: {value:.4f}"
        )


if __name__ == "__main__":

    main()