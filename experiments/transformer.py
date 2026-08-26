import os
import sys
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    classification_report,
    ConfusionMatrixDisplay,
)

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# CONFIG
# ============================================================

CHBMIT_DIR = r"C:\Datasets\chb_mit"
SIENA_DIR = r"C:\Datasets\siena_processed"
BONN_SPLIT_DIR = r"C:\Datasets\bonn_splits\bonn_splits"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42

BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 8

WEIGHT_DECAY = 1e-4
DROPOUT = 0.4

MODEL_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "transformer"
)

TRANSFORMER_LR = 3e-4
WARMUP_EPOCHS = 5

N_HEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128

print(f"Using device: {DEVICE}")


# ============================================================
# SHARED CNN ENCODER
# ============================================================

class ChannelAgnosticEncoder(nn.Module):

    def __init__(self, feature_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                1,
                32,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(
                64,
                feature_dim,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):

        # x = (batch, channels, time)
        b, c, t = x.shape

        x = x.reshape(
            b * c,
            1,
            t
        )

        feat = self.net(x)

        _, fd, rt = feat.shape

        feat = feat.reshape(
            b,
            c,
            fd,
            rt
        )

        # channel pooling
        return feat.max(dim=1).values


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=2000
    ):
        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[
            :, :x.size(1), :
        ]


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class TransformerClassifier(nn.Module):

    def __init__(
        self,
        feature_dim=64,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ):

        super().__init__()

        self.encoder = ChannelAgnosticEncoder(
            feature_dim
        )

        self.pos_encoding = PositionalEncoding(
            feature_dim
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                feature_dim,
                32
            ),
            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                32,
                1
            ),
        )

    def forward(self, x):

        # CNN feature extraction
        seq = self.encoder(x)

        # (B, feature_dim, time)
        # ->
        # (B, time, feature_dim)

        seq = seq.transpose(1, 2)

        seq = self.pos_encoding(seq)

        seq = self.transformer(seq)

        pooled = seq.mean(dim=1)

        return self.classifier(
            pooled
        ).squeeze(-1)


# ============================================================
# MEMORY-EFFICIENT NPZ DATASET
# ============================================================

class NPZEEGDataset(Dataset):

    """
    Lazy-loading dataset.

    IMPORTANT:
    Does NOT concatenate all EEG windows into RAM.

    Each sample is loaded from the corresponding NPZ file
    only when requested by the DataLoader.
    """

    def __init__(
        self,
        samples
    ):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        file_path, window_type, window_idx = self.samples[idx]

        with np.load(
            file_path,
            allow_pickle=True
        ) as d:

            scale = float(
                d["scale_factor"]
            )

            if scale == 0:
                raise ValueError(
                    f"Invalid scale_factor=0 in {file_path}"
                )

            if window_type == "seizure":

                x = d[
                    "seizure_windows"
                ][window_idx]

                y = 1.0

            else:

                x = d[
                    "normal_windows"
                ][window_idx]

                y = 0.0

        x = x.astype(
            np.float32
        ) / scale

        return (
            torch.from_numpy(x),
            torch.tensor(
                y,
                dtype=torch.float32
            )
        )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_bonn(X):

    # Bonn is already normalized during preprocessing.
    return X


# ============================================================
# DISCOVER SUBJECTS
# ============================================================

def discover_subjects(
    data_dir,
    subject_prefix
):

    if not os.path.isdir(data_dir):

        raise FileNotFoundError(
            f"Dataset directory not found:\n{data_dir}"
        )

    subject_dirs = []

    for root, dirs, files in os.walk(
        data_dir
    ):

        if any(
            f.lower().endswith(".npz")
            for f in files
        ):

            rel = os.path.relpath(
                root,
                data_dir
            )

            subject_name = os.path.basename(
                rel
            )

            if subject_name.lower().startswith(
                subject_prefix.lower()
            ):

                subject_dirs.append(
                    rel
                )

    subjects = sorted(
        set(subject_dirs)
    )

    return subjects


# ============================================================
# PATIENT-WISE SPLIT
# ============================================================

def make_subject_split(
    subjects,
    val_frac=1 / 6,
    test_frac=1 / 6
):

    n = len(subjects)

    n_test = max(
        1,
        round(n * test_frac)
    )

    n_val = max(
        1,
        round(n * val_frac)
    )

    test_subjects = subjects[
        -n_test:
    ]

    val_subjects = subjects[
        -(n_test + n_val):-n_test
    ]

    train_subjects = [
        s
        for s in subjects
        if s not in test_subjects
        and s not in val_subjects
    ]

    return (
        train_subjects,
        val_subjects,
        test_subjects
    )


# ============================================================
# BUILD LAZY SAMPLE INDEX
# ============================================================

def build_sample_index(
    data_dir,
    subjects,
    required_channels=None
):

    samples = []

    n_channels = None

    for subject in subjects:

        pattern = os.path.join(
            data_dir,
            subject,
            "*.npz"
        )

        files = sorted(
            glob.glob(pattern)
        )

        for file_path in files:

            with np.load(
                file_path,
                allow_pickle=True
            ) as d:

                selected_channels = d[
                    "selected_channels"
                ]

                file_channels = len(
                    selected_channels
                )

                if required_channels is None:

                    required_channels = file_channels
                    n_channels = file_channels

                if file_channels != required_channels:
                    continue

                n_seizure = d[
                    "seizure_windows"
                ].shape[0]

                n_normal = d[
                    "normal_windows"
                ].shape[0]

            for i in range(n_seizure):

                samples.append(
                    (
                        file_path,
                        "seizure",
                        i
                    )
                )

            for i in range(n_normal):

                samples.append(
                    (
                        file_path,
                        "normal",
                        i
                    )
                )

    if not samples:

        raise ValueError(
            "No usable samples found."
        )

    return samples, n_channels


# ============================================================
# LOAD CHB-MIT / SIENA
# ============================================================

def load_npz_dataset(
    data_dir,
    subject_prefix
):

    subjects = discover_subjects(
        data_dir,
        subject_prefix
    )

    if len(subjects) < 3:

        raise ValueError(
            f"Only {len(subjects)} subjects found:\n"
            f"{subjects}"
        )

    (
        train_subjects,
        val_subjects,
        test_subjects
    ) = make_subject_split(
        subjects
    )

    print(
        f"Detected {len(subjects)} "
        f"{subject_prefix} subjects:"
    )

    print(subjects)

    print(
        f"Patient-wise split: "
        f"{len(train_subjects)} train / "
        f"{len(val_subjects)} validation / "
        f"{len(test_subjects)} test"
    )

    print(
        f"Train subjects: "
        f"{train_subjects}"
    )

    print(
        f"Validation subjects: "
        f"{val_subjects}"
    )

    print(
        f"Test subjects: "
        f"{test_subjects}"
    )

    # Determine common channel count
    train_samples, n_channels = build_sample_index(
        data_dir,
        train_subjects
    )

    val_samples, _ = build_sample_index(
        data_dir,
        val_subjects,
        required_channels=n_channels
    )

    test_samples, _ = build_sample_index(
        data_dir,
        test_subjects,
        required_channels=n_channels
    )

    print(
        f"Channels used: {n_channels}"
    )

    print(
        f"Train samples: "
        f"{len(train_samples):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_samples):,}"
    )

    print(
        f"Test samples: "
        f"{len(test_samples):,}"
    )

    return (
        train_samples,
        val_samples,
        test_samples,
        n_channels
    )


def load_chbmit():

    return load_npz_dataset(
        CHBMIT_DIR,
        "chb"
    )


def load_siena():

    return load_npz_dataset(
        SIENA_DIR,
        "PN"
    )


# ============================================================
# BONN LOADER
# ============================================================

class BonnDataset(Dataset):

    def __init__(
        self,
        X_path,
        y_path
    ):

        self.X = np.load(
            X_path,
            mmap_mode="r"
        )

        self.y = np.load(
            y_path,
            mmap_mode="r"
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        x = np.asarray(
            self.X[idx],
            dtype=np.float32
        )

        # Add channel dimension
        x = x[None, :]

        y = np.float32(
            self.y[idx]
        )

        return (
            torch.from_numpy(x.copy()),
            torch.tensor(
                y,
                dtype=torch.float32
            )
        )


def load_bonn():

    train_dataset = BonnDataset(
        os.path.join(
            BONN_SPLIT_DIR,
            "X_train.npy"
        ),
        os.path.join(
            BONN_SPLIT_DIR,
            "y_train.npy"
        )
    )

    val_dataset = BonnDataset(
        os.path.join(
            BONN_SPLIT_DIR,
            "X_val.npy"
        ),
        os.path.join(
            BONN_SPLIT_DIR,
            "y_val.npy"
        )
    )

    test_dataset = BonnDataset(
        os.path.join(
            BONN_SPLIT_DIR,
            "X_test.npy"
        ),
        os.path.join(
            BONN_SPLIT_DIR,
            "y_test.npy"
        )
    )

    print(
        f"Train: {len(train_dataset):,} windows"
    )

    print(
        f"Val: {len(val_dataset):,} windows"
    )

    print(
        f"Test: {len(test_dataset):,} windows"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        1
    )


# ============================================================
# METRICS
# ============================================================

def youden_threshold(
    y_true,
    probs
):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        probs
    )

    finite = np.isfinite(
        thresholds
    )

    if not finite.any():

        return 0.5

    j = (
        tpr[finite]
        - fpr[finite]
    )

    return float(
        thresholds[finite][
            np.argmax(j)
        ]
    )


def metrics_at(
    y_true,
    probs,
    threshold
):

    preds = (
        probs >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        preds,
        labels=[0, 1]
    ).ravel()

    return {

        "accuracy":
            accuracy_score(
                y_true,
                preds
            ),

        "precision":
            precision_score(
                y_true,
                preds,
                zero_division=0
            ),

        "sensitivity":
            tp / (tp + fn)
            if (tp + fn) > 0
            else float("nan"),

        "specificity":
            tn / (tn + fp)
            if (tn + fp) > 0
            else float("nan"),

        "f1":
            f1_score(
                y_true,
                preds,
                zero_division=0
            ),

        "confusion":
            (tn, fp, fn, tp)
    }


# ============================================================
# GET PROBABILITIES
# ============================================================

def get_probs(
    model,
    loader
):

    model.eval()

    probs = []
    labels = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            p = torch.sigmoid(
                model(xb)
            )

            probs.extend(
                p.cpu().numpy()
            )

            labels.extend(
                yb.numpy()
            )

    return (
        np.asarray(probs),
        np.asarray(labels)
    )


# ============================================================
# LR WARMUP
# ============================================================

def make_warmup_scheduler(
    optimizer,
    warmup_epochs
):

    def lr_lambda(epoch):

        if epoch < warmup_epochs:

            return (
                (epoch + 1)
                / warmup_epochs
            )

        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def run_dataset(
    name,
    loader_fn
):

    print(
        "\n"
        + "#" * 60
    )

    print(
        f"# DATASET: {name}"
    )

    print(
        "#" * 60
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    loaded = loader_fn()

    # --------------------------------------------------------
    # BONN
    # --------------------------------------------------------

    if name.lower() == "bonn":

        train_dataset = loaded[0]
        val_dataset = loaded[1]
        test_dataset = loaded[2]

        n_channels = loaded[3]

        print(
            "Normalization: "
            "Bonn preprocessing already normalized."
        )

    # --------------------------------------------------------
    # CHB-MIT / SIENA
    # --------------------------------------------------------

    else:

        train_samples = loaded[0]
        val_samples = loaded[1]
        test_samples = loaded[2]

        n_channels = loaded[3]

        train_dataset = NPZEEGDataset(
            train_samples
        )

        val_dataset = NPZEEGDataset(
            val_samples
        )

        test_dataset = NPZEEGDataset(
            test_samples
        )

        print(
            "Normalization: "
            "using preprocessing scale_factor."
        )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"Channels: {n_channels}"
    )

    print(
        f"Train: {len(train_dataset):,} windows"
    )

    print(
        f"Val: {len(val_dataset):,} windows"
    )

    print(
        f"Test: {len(test_dataset):,} windows"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = TransformerClassifier().to(
        DEVICE
    )

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    if name.lower() == "bonn":

        y_train = np.asarray(
            train_dataset.y
        )

        n_pos = y_train.sum()

        n_neg = (
            len(y_train)
            - n_pos
        )

    else:

        n_pos = sum(
            1
            for s in train_dataset.samples
            if s[1] == "seizure"
        )

        n_neg = (
            len(train_dataset)
            - n_pos
        )

    pos_weight = torch.tensor(
        [
            n_neg / max(
                n_pos,
                1
            )
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRANSFORMER_LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = make_warmup_scheduler(
        optimizer,
        WARMUP_EPOCHS
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_auroc = -np.inf
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(
        MAX_EPOCHS
    ):

        model.train()

        train_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(
                DEVICE,
                non_blocking=True
            )

            yb = yb.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * xb.size(0)
            )

        scheduler.step()

        train_loss /= len(
            train_loader.dataset
        )

        # Validation
        val_probs, val_labels = get_probs(
            model,
            val_loader
        )

        val_auroc = roc_auc_score(
            val_labels,
            val_probs
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"[{name}] "
            f"Epoch {epoch + 1:>3} | "
            f"lr: {current_lr:.2e} | "
            f"train loss: {train_loss:.4f} | "
            f"val AUROC: {val_auroc:.4f}"
        )

        if val_auroc > best_val_auroc:

            best_val_auroc = val_auroc

            best_state = {
                k:
                v.detach()
                .cpu()
                .clone()

                for k, v
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= PATIENCE
            ):

                print(
                    f"[{name}] "
                    f"Early stopping at "
                    f"epoch {epoch + 1}"
                )

                break

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            f"{name}: "
            "no valid checkpoint produced."
        )

    model.load_state_dict(
        best_state
    )

    print(
        f"[{name}] "
        f"Best validation AUROC: "
        f"{best_val_auroc:.4f}"
    )

    # --------------------------------------------------------
    # FINAL PROBABILITIES
    # --------------------------------------------------------

    train_probs, train_labels = get_probs(
        model,
        train_loader
    )

    val_probs, val_labels = get_probs(
        model,
        val_loader
    )

    test_probs, test_labels = get_probs(
        model,
        test_loader
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    threshold = youden_threshold(
        val_labels,
        val_probs
    )

    train_result = metrics_at(
        train_labels,
        train_probs,
        threshold
    )

    test_result = metrics_at(
        test_labels,
        test_probs,
        threshold
    )

    auroc = roc_auc_score(
        test_labels,
        test_probs
    )

    auprc = average_precision_score(
        test_labels,
        test_probs
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 55
    )

    print(
        f"RESULTS — Transformer — {name}"
    )

    print(
        "=" * 55
    )

    print(
        f"AUROC:     {auroc:.3f}"
    )

    print(
        f"AUPRC:     {auprc:.3f}"
    )

    print(
        f"Threshold: {threshold:.3f}"
    )

    print()

    print(
        f"{'Metric':<15}"
        f"{'Train':<12}"
        f"{'Test':<12}"
    )

    print(
        f"{'Accuracy':<15}"
        f"{train_result['accuracy']:<12.3f}"
        f"{test_result['accuracy']:.3f}"
    )

    print(
        f"{'Precision':<15}"
        f"{train_result['precision']:<12.3f}"
        f"{test_result['precision']:.3f}"
    )

    print(
        f"{'Sensitivity':<15}"
        f"{train_result['sensitivity']:<12.3f}"
        f"{test_result['sensitivity']:.3f}"
    )

    print(
        f"{'Specificity':<15}"
        f"{train_result['specificity']:<12.3f}"
        f"{test_result['specificity']:.3f}"
    )

    print(
        f"{'F1':<15}"
        f"{train_result['f1']:<12.3f}"
        f"{test_result['f1']:.3f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    test_preds = (
        test_probs >= threshold
    ).astype(int)

    print(
        f"\nClassification report — "
        f"{name}"
    )

    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=[
                "Normal",
                "Seizure"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    test_cm = confusion_matrix(
        test_labels,
        test_preds,
        labels=[0, 1]
    )

    print(
        f"Test confusion matrix - "
        f"{name}:\n{test_cm}"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=test_cm,
        display_labels=[
            "Normal",
            "Seizure"
        ]
    )

    display.plot(
        cmap="Blues",
        values_format="d",
        colorbar=True
    )

    plt.title(
        f"Confusion Matrix - "
        f"Transformer - {name}"
    )

    plt.tight_layout()

    os.makedirs(
        MODEL_OUTPUT_DIR,
        exist_ok=True
    )

    confusion_path = os.path.join(
        MODEL_OUTPUT_DIR,
        f"transformer_"
        f"{name.lower().replace('-', '_')}"
        f"_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved confusion matrix: "
        f"{confusion_path}"
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        MODEL_OUTPUT_DIR,
        f"transformer_"
        f"{name.lower().replace('-', '_')}"
        f"_best.pt"
    )

    torch.save(
        {
            "dataset": name,

            "model_state_dict":
                best_state,

            "threshold":
                float(threshold),

            "threshold_method":
                "Youden's J on validation set",

            "best_validation_auroc":
                float(best_val_auroc),

            "test_auroc":
                float(auroc),

            "test_auprc":
                float(auprc),

            "test_metrics":
                test_result,

            "n_channels":
                int(n_channels),

            "hyperparams": {
                "lr":
                    TRANSFORMER_LR,

                "warmup_epochs":
                    WARMUP_EPOCHS,

                "nhead":
                    N_HEAD,

                "num_layers":
                    NUM_LAYERS,

                "dim_feedforward":
                    DIM_FEEDFORWARD,

                "dropout":
                    DROPOUT,

                "weight_decay":
                    WEIGHT_DECAY
            }
        },
        checkpoint_path
    )

    print(
        f"Saved model: "
        f"{checkpoint_path}"
    )

    return {

        "dataset":
            name,

        "model":
            "Transformer",

        "auroc":
            auroc,

        "auprc":
            auprc,

        "precision":
            test_result["precision"],

        "sensitivity":
            test_result["sensitivity"],

        "specificity":
            test_result["specificity"],

        "f1":
            test_result["f1"],

        "accuracy_test":
            test_result["accuracy"]
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    experiments = [
        ("Bonn", load_bonn),
        ("CHB-MIT", load_chbmit),
        ("Siena", load_siena)
    ]

    results = []

    for dataset_name, loader_fn in experiments:

        results.append(
            run_dataset(
                dataset_name,
                loader_fn
            )
        )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "FINAL COMPARISON TABLE — "
        "Transformer across all 3 datasets"
    )

    print(
        "=" * 70
    )

    header = (
        f"{'Dataset':<10}"
        f"{'Model':<12}"
        f"{'AUROC':<8}"
        f"{'AUPRC':<8}"
        f"{'Prec':<8}"
        f"{'Sens':<8}"
        f"{'Spec':<8}"
        f"{'F1':<8}"
    )

    print(header)

    print(
        "-" * len(header)
    )

    for r in results:

        print(
            f"{r['dataset']:<10}"
            f"{r['model']:<12}"
            f"{r['auroc']:<8.3f}"
            f"{r['auprc']:<8.3f}"
            f"{r['precision']:<8.3f}"
            f"{r['sensitivity']:<8.3f}"
            f"{r['specificity']:<8.3f}"
            f"{r['f1']:<8.3f}"
        )