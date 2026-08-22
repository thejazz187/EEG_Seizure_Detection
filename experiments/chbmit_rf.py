
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, confusion_matrix, roc_curve,
)

from models.eeg_features_chbmit import extract_features_batch
CHBMIT_DIR = r"C:\Users\hsbho\Downloads\Major Project\Datasets\chbmit_processed"

def load_chbmit():
    subjects = sorted(d for d in os.listdir(CHBMIT_DIR)
                       if os.path.isdir(os.path.join(CHBMIT_DIR, d)) and d.startswith("chb"))
    test_subj, val_subj = subjects[-4:], subjects[-8:-4]
    train_subj = [s for s in subjects if s not in test_subj and s not in val_subj]

    counts = {}
    for subj in subjects:
        for f in glob.glob(os.path.join(CHBMIT_DIR, subj, "*.npz")):
            n = len(list(np.load(f, allow_pickle=True)["selected_channels"]))
            counts[n] = counts.get(n, 0) + 1
    n_channels = max(counts, key=counts.get)

    def load(subj_list):
        X, y = [], []
        for subject in subj_list:
            for f in glob.glob(os.path.join(CHBMIT_DIR, subject, "*.npz")):
                d = np.load(f, allow_pickle=True)
                if len(list(d["selected_channels"])) != n_channels:
                    continue
                scale = float(d["scale_factor"])
                s, n = d["seizure_windows"] / scale, d["normal_windows"] / scale
                if s.shape[0]:
                    X.append(s); y.append(np.ones(s.shape[0]))
                if n.shape[0]:
                    X.append(n); y.append(np.zeros(n.shape[0]))
        return np.concatenate(X).astype(np.float32), np.concatenate(y)

    return load(train_subj), load(val_subj), load(test_subj), n_channels, (train_subj, val_subj, test_subj)


def balanced_threshold(y_true, probs):
    """Youden's J statistic: maximizes (sensitivity + specificity - 1) —
    the best overall tradeoff point, rather than pushing sensitivity to an
    extreme at specificity's expense."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return thresholds[np.argmax(j)]


def metrics_at(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, preds),
        "confusion": (tn, fp, fn, tp),
    }

(X_train, y_train), (X_val, y_val), (X_test, y_test), n_channels, subjects = load_chbmit()
train_subj, val_subj, test_subj = subjects

print(f"CHB-MIT | {len(train_subj)} train / {len(val_subj)} val / {len(test_subj)} test subjects, {n_channels} channels")
print(f"Train: {X_train.shape[0]:,} windows  |  Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

print("Extracting train features...")
X_train_f = extract_features_batch(X_train)
print("Extracting val features...")
X_val_f = extract_features_batch(X_val)
print("Extracting test features...")
X_test_f = extract_features_batch(X_test)
print(f"Feature vector size: {X_train_f.shape[1]}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_f, y_train)

train_probs = rf.predict_proba(X_train_f)[:, 1]
val_probs = rf.predict_proba(X_val_f)[:, 1]
test_probs = rf.predict_proba(X_test_f)[:, 1]

threshold = balanced_threshold(y_val, val_probs)

train_result = metrics_at(y_train, train_probs, threshold)
test_result = metrics_at(y_test, test_probs, threshold)
auroc = roc_auc_score(y_test, test_probs)
auprc = average_precision_score(y_test, test_probs)

print("=" * 55)
print("RESULTS — CHB-MIT")
print("=" * 55)
print(f"AUROC:        {auroc:.3f}")
print(f"AUPRC:        {auprc:.3f}")
print(f"Threshold:    {threshold:.3f}  (Youden's J, chosen on validation set)")
print()
print(f"{'Metric':<15}{'Train':<12}{'Test':<12}")
print(f"{'Accuracy':<15}{train_result['accuracy']:<12.3f}{test_result['accuracy']:.3f}")
print(f"{'Sensitivity':<15}{train_result['sensitivity']:<12.3f}{test_result['sensitivity']:.3f}")
print(f"{'Specificity':<15}{train_result['specificity']:<12.3f}{test_result['specificity']:.3f}")
print(f"{'F1':<15}{train_result['f1']:<12.3f}{test_result['f1']:.3f}")

tn, fp, fn, tp = test_result["confusion"]
cm = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix (Test)")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Seizure"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Seizure"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("chbmit_confusion_matrix.png", dpi=150)
plt.show()

fpr, tpr, _ = roc_curve(y_test, test_probs)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="crimson", label=f"AUROC = {auroc:.3f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("ROC Curve (Test)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
plt.tight_layout()
plt.savefig("chbmit_roc_curve.png", dpi=150)
plt.show()