"""
Shared EEG feature extraction for the Random Forest baseline.

Input:
    Single-channel Bonn window: (1024,)
    Multi-channel EEG window:   (n_channels, 1024)

Output:
    Fixed 69-dimensional feature vector.

23 features per channel:
    Time domain      : 9
    Frequency domain: 11
    Hjorth           : 3

23 × 3 aggregations (mean, std, max) = 69 features.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

FS = 256

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}


# NumPy compatibility
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# HJORTH PARAMETERS
# ============================================================

def _hjorth_params(x):
    """
    Calculate Hjorth activity, mobility and complexity.
    """

    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    activity = var_x

    mobility = (
        np.sqrt(var_dx / var_x)
        if var_x > 0
        else 0.0
    )

    mobility_dx = (
        np.sqrt(var_ddx / var_dx)
        if var_dx > 0
        else 0.0
    )

    complexity = (
        mobility_dx / mobility
        if mobility > 0
        else 0.0
    )

    return activity, mobility, complexity


# ============================================================
# FREQUENCY FEATURES
# ============================================================

def _band_powers(x, fs=FS):
    """
    Calculate absolute and relative power for:
        delta, theta, alpha, beta, gamma

    Also calculates normalized spectral entropy.
    """

    freqs, psd = signal.welch(
        x,
        fs=fs,
        nperseg=min(256, len(x))
    )

    # Total power
    if len(freqs) > 1:
        total_power = _trapz(
            psd,
            freqs
        )
    else:
        total_power = np.sum(psd)

    if total_power <= 0:
        total_power = 1e-12

    abs_powers = []
    rel_powers = []

    for low, high in BANDS.values():

        mask = (
            (freqs >= low)
            &
            (freqs <= high)
        )

        if mask.sum() > 1:

            band_power = _trapz(
                psd[mask],
                freqs[mask]
            )

        else:
            band_power = 0.0

        abs_powers.append(
            band_power
        )

        rel_powers.append(
            band_power / total_power
        )

    # --------------------------------------------------------
    # Spectral entropy
    # --------------------------------------------------------

    psd_norm = (
        psd /
        (psd.sum() + 1e-12)
    )

    psd_norm = psd_norm[
        psd_norm > 0
    ]

    if len(psd_norm) > 1:

        spectral_entropy = -np.sum(
            psd_norm *
            np.log2(psd_norm)
        )

        spectral_entropy /= np.log2(
            len(psd_norm)
        )

    else:
        spectral_entropy = 0.0

    return (
        abs_powers,
        rel_powers,
        spectral_entropy
    )


# ============================================================
# FEATURES FOR ONE CHANNEL
# ============================================================

def _channel_features(x, fs=FS):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Time-domain features
    # --------------------------------------------------------

    mean = np.mean(x)

    std = np.std(x)

    variance = np.var(x)

    skewness = skew(x)

    kurt = kurtosis(x)

    line_length = np.sum(
        np.abs(np.diff(x))
    )

    zero_crossings = np.sum(
        np.diff(np.sign(x)) != 0
    )

    zero_crossing_rate = (
        zero_crossings / len(x)
    )

    peak_to_peak = np.ptp(x)

    rms = np.sqrt(
        np.mean(x ** 2)
    )

    # --------------------------------------------------------
    # Hjorth parameters
    # --------------------------------------------------------

    (
        hjorth_activity,
        hjorth_mobility,
        hjorth_complexity
    ) = _hjorth_params(x)

    # --------------------------------------------------------
    # Frequency-domain features
    # --------------------------------------------------------

    (
        abs_powers,
        rel_powers,
        spectral_entropy
    ) = _band_powers(
        x,
        fs=fs
    )

    # --------------------------------------------------------
    # Final 23 features
    # --------------------------------------------------------

    features = [

        # Time domain — 9
        mean,
        std,
        variance,
        skewness,
        kurt,
        line_length,
        zero_crossing_rate,
        peak_to_peak,
        rms,

        # Hjorth — 3
        hjorth_activity,
        hjorth_mobility,
        hjorth_complexity,

        # Spectral entropy — 1
        spectral_entropy,

        # Absolute band powers — 5
        *abs_powers,

        # Relative band powers — 5
        *rel_powers,
    ]

    return np.array(
        features,
        dtype=np.float64
    )


# ============================================================
# FEATURE NAMES
# ============================================================

CHANNEL_FEATURE_NAMES = (

    [
        "mean",
        "std",
        "var",
        "skew",
        "kurtosis",
        "line_length",
        "zero_crossing_rate",
        "peak_to_peak",
        "rms",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "spectral_entropy",
    ]

    +

    [
        f"abs_power_{band}"
        for band in BANDS
    ]

    +

    [
        f"rel_power_{band}"
        for band in BANDS
    ]
)


# ============================================================
# FEATURES FOR ONE EEG WINDOW
# ============================================================

def extract_window_features(
    window,
    fs=FS
):
    """
    Input:
        (1024,)              -> Bonn
        (n_channels, 1024)   -> CHB-MIT / Siena

    Output:
        69-dimensional vector
    """

    window = np.asarray(
        window
    )

    # --------------------------------------------------------
    # Bonn single-channel EEG
    # --------------------------------------------------------

    if window.ndim == 1:

        window = window[
            np.newaxis,
            :
        ]

    # --------------------------------------------------------
    # Extract features channel-by-channel
    # --------------------------------------------------------

    per_channel = np.stack(
        [
            _channel_features(
                window[ch],
                fs=fs
            )

            for ch in range(
                window.shape[0]
            )
        ]
    )

    # Shape:
    # (number_of_channels, 23)

    # --------------------------------------------------------
    # Aggregate across channels
    # --------------------------------------------------------

    agg_mean = per_channel.mean(
        axis=0
    )

    agg_std = per_channel.std(
        axis=0
    )

    agg_max = per_channel.max(
        axis=0
    )

    # --------------------------------------------------------
    # 23 × 3 = 69
    # --------------------------------------------------------

    features = np.concatenate(
        [
            agg_mean,
            agg_std,
            agg_max
        ]
    )

    return features


# ============================================================
# FEATURE NAMES FOR 69-D VECTOR
# ============================================================

def feature_names():

    names = []

    for aggregation in [
        "mean",
        "std",
        "max"
    ]:

        names += [
            f"{aggregation}_{name}"
            for name
            in CHANNEL_FEATURE_NAMES
        ]

    return names


# ============================================================
# BATCH FEATURE EXTRACTION
# ============================================================

def extract_features_batch(
    windows,
    fs=FS
):
    """
    Input:
        Bonn:
            (n_windows, 1024)

        Multi-channel:
            (n_windows, n_channels, 1024)

    Output:
        (n_windows, 69)
    """

    windows = np.asarray(
        windows
    )

    features = np.stack(
        [
            extract_window_features(
                window,
                fs=fs
            )

            for window in windows
        ]
    )

    return features