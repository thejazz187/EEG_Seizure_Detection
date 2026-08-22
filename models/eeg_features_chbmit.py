"""
Vectorized version of eeg_features.py — same features, same output shape,
much faster. The original looped window-by-window, channel-by-channel,
calling scipy.signal.welch individually for each one (this is what was
slow). This version computes every feature across many signals at once.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

FS = 256

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

CHANNEL_FEATURE_NAMES = (
    ["mean", "std", "var", "skew", "kurtosis",
     "line_length", "zero_crossing_rate", "peak_to_peak", "rms",
     "hjorth_activity", "hjorth_mobility", "hjorth_complexity", "spectral_entropy"]
    + [f"abs_power_{b}" for b in BANDS]
    + [f"rel_power_{b}" for b in BANDS]
)


def feature_names():
    names = []
    for agg in ("mean", "std", "max"):
        names += [f"{agg}_{n}" for n in CHANNEL_FEATURE_NAMES]
    return names


def _channel_features_vectorized(chunk, fs=FS):
    """chunk: (n_signals, n_samples) -> (n_signals, 23) — all channel-level
    features computed for every signal in the chunk at once."""
    mean = chunk.mean(axis=1)
    std = chunk.std(axis=1)
    var = chunk.var(axis=1)
    sk = skew(chunk, axis=1)
    ku = kurtosis(chunk, axis=1)

    diffs = np.diff(chunk, axis=1)
    line_length = np.abs(diffs).sum(axis=1)
    zcr = (np.diff(np.sign(chunk), axis=1) != 0).sum(axis=1) / chunk.shape[1]
    ptp = chunk.max(axis=1) - chunk.min(axis=1)
    rms = np.sqrt((chunk ** 2).mean(axis=1))

    diff2 = np.diff(diffs, axis=1)
    var_x, var_dx, var_ddx = chunk.var(axis=1), diffs.var(axis=1), diff2.var(axis=1)
    activity = var_x
    mobility = np.sqrt(np.where(var_x > 0, var_dx / np.where(var_x > 0, var_x, 1), 0))
    mobility_dx = np.sqrt(np.where(var_dx > 0, var_ddx / np.where(var_dx > 0, var_dx, 1), 0))
    complexity = np.where(mobility > 0, mobility_dx / np.where(mobility > 0, mobility, 1), 0)

    freqs, psd = signal.welch(chunk, fs=fs, nperseg=min(256, chunk.shape[1]), axis=1)
    total_power = trapezoid(psd, freqs, axis=1)
    total_power = np.where(total_power > 0, total_power, 1e-12)

    abs_powers, rel_powers = [], []
    for low, high in BANDS.values():
        mask = (freqs >= low) & (freqs <= high)
        bp = trapezoid(psd[:, mask], freqs[mask], axis=1) if mask.sum() > 1 else np.zeros(chunk.shape[0])
        abs_powers.append(bp)
        rel_powers.append(bp / total_power)
    abs_powers = np.stack(abs_powers, axis=1)
    rel_powers = np.stack(rel_powers, axis=1)

    psd_sum = psd.sum(axis=1, keepdims=True) + 1e-12
    psd_norm = psd / psd_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(psd_norm > 0, psd_norm * np.log2(psd_norm), 0)
    spectral_entropy = -entropy_terms.sum(axis=1)
    n_freqs = psd.shape[1]
    spectral_entropy = spectral_entropy / (np.log2(n_freqs) if n_freqs > 1 else 1.0)

    return np.column_stack([
        mean, std, var, sk, ku,
        line_length, zcr, ptp, rms,
        activity, mobility, complexity,
        spectral_entropy,
        abs_powers, rel_powers,
    ])


def extract_features_batch(windows, fs=FS, chunk_size=5000, verbose=True):
    """windows: (n_windows, n_channels, n_samples) or (n_windows, n_samples)."""
    if windows.ndim == 2:
        windows = windows[:, np.newaxis, :]

    n_windows, n_channels, n_samples = windows.shape
    flat = windows.reshape(n_windows * n_channels, n_samples)
    total = flat.shape[0]

    per_channel_chunks = []
    for start in range(0, total, chunk_size):
        chunk = flat[start:start + chunk_size]
        per_channel_chunks.append(_channel_features_vectorized(chunk, fs=fs))
        end = start + chunk.shape[0]
        if verbose and (end // 100000 != start // 100000 or end == total):
            print(f"  {end:,}/{total:,} signals processed")

    per_channel = np.concatenate(per_channel_chunks, axis=0)   # (n_windows*n_channels, 23)
    per_channel = per_channel.reshape(n_windows, n_channels, -1)

    agg_mean = per_channel.mean(axis=1)
    agg_std = per_channel.std(axis=1)
    agg_max = per_channel.max(axis=1)

    return np.concatenate([agg_mean, agg_std, agg_max], axis=1)
