"""
RF signal preprocessing utilities.

Covers the standard IQ -> spectrogram pipeline used throughout SpectraNet:
  raw IQ  ->  normalization  ->  STFT / FFT  ->  spectrogram (log-magnitude)
  ->  per-sample normalization  ->  tensor ready for a CNN.

All functions operate on NumPy arrays and are deliberately dependency-light
(NumPy only) so they can be reused inside a PyTorch Dataset, a notebook,
or a standalone offline ETL job without dragging in torch.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# IQ normalization
# --------------------------------------------------------------------------- #

def normalize_iq(iq: np.ndarray, mode: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    """
    Normalize a complex or 2xN real IQ array.

    Parameters
    ----------
    iq : np.ndarray
        Either complex-valued shape (N,), or real-valued shape (2, N) with
        row 0 = I, row 1 = Q.
    mode : {"zscore", "unit_energy", "peak"}
        zscore      -> zero mean, unit variance per-channel
        unit_energy -> scale so mean(|x|^2) == 1 (classic RF normalization)
        peak        -> scale so max(|x|) == 1
    eps : float
        Numerical stability floor.

    Returns
    -------
    np.ndarray
        Normalized array, same shape/dtype family as input.
    """
    is_complex = np.iscomplexobj(iq)
    x = iq if is_complex else iq[0] + 1j * iq[1]

    if mode == "zscore":
        mu = x.mean()
        sigma = x.std() + eps
        x = (x - mu) / sigma
    elif mode == "unit_energy":
        energy = np.mean(np.abs(x) ** 2) + eps
        x = x / np.sqrt(energy)
    elif mode == "peak":
        peak = np.max(np.abs(x)) + eps
        x = x / peak
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    if is_complex:
        return x
    return np.stack([x.real, x.imag], axis=0)


# --------------------------------------------------------------------------- #
# Spectral transforms
# --------------------------------------------------------------------------- #

def compute_fft(iq: np.ndarray, n_fft: int | None = None, shift: bool = True) -> np.ndarray:
    """Single-frame FFT magnitude spectrum of a complex IQ vector."""
    x = iq if np.iscomplexobj(iq) else iq[0] + 1j * iq[1]
    n_fft = n_fft or len(x)
    spec = np.fft.fft(x, n=n_fft)
    if shift:
        spec = np.fft.fftshift(spec)
    return spec


def compute_stft(
    iq: np.ndarray,
    n_fft: int = 128,
    hop_length: int = 32,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """
    Short-time Fourier transform of a complex IQ signal, implemented with
    plain NumPy (no torch/scipy dependency required at preprocessing time).

    Returns
    -------
    np.ndarray, complex, shape (n_fft, n_frames)
    """
    x = iq if np.iscomplexobj(iq) else iq[0] + 1j * iq[1]

    if window == "hann":
        win = np.hanning(n_fft)
    elif window == "hamming":
        win = np.hamming(n_fft)
    elif window == "rect" or window is None:
        win = np.ones(n_fft)
    else:
        raise ValueError(f"Unknown window: {window}")

    if center:
        pad = n_fft // 2
        x = np.pad(x, (pad, pad), mode="reflect")

    n_frames = 1 + (len(x) - n_fft) // hop_length
    if n_frames < 1:
        raise ValueError("Signal too short for given n_fft/hop_length.")

    frames = np.empty((n_fft, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop_length
        seg = x[start:start + n_fft] * win
        frames[:, i] = np.fft.fftshift(np.fft.fft(seg, n=n_fft))

    return frames


def spectrogram(
    iq: np.ndarray,
    n_fft: int = 128,
    hop_length: int = 32,
    window: str = "hann",
    log_scale: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Full IQ -> log-magnitude spectrogram pipeline.

    Returns
    -------
    np.ndarray, float32, shape (n_fft, n_frames)
        Log-magnitude (dB-like) spectrogram, normalized to roughly [0, 1]
        via min-max scaling. Ready to feed to a CNN as a single-channel
        (or stacked with phase as a 2-channel) image.
    """
    stft = compute_stft(iq, n_fft=n_fft, hop_length=hop_length, window=window)
    mag = np.abs(stft)

    if log_scale:
        mag = np.log10(mag + eps)

    mag = mag.astype(np.float32)
    mag -= mag.min()
    denom = mag.max() + eps
    mag /= denom
    return mag


def spectrogram_with_phase(
    iq: np.ndarray,
    n_fft: int = 128,
    hop_length: int = 32,
    window: str = "hann",
) -> np.ndarray:
    """
    Two-channel representation: [log-magnitude, unwrapped-phase-derivative].
    Often gives CNNs extra discriminative signal for modulation classification.

    Returns
    -------
    np.ndarray, float32, shape (2, n_fft, n_frames)
    """
    stft = compute_stft(iq, n_fft=n_fft, hop_length=hop_length, window=window)
    mag = np.log10(np.abs(stft) + 1e-10).astype(np.float32)
    mag -= mag.min()
    mag /= (mag.max() + 1e-10)

    phase = np.angle(stft)
    phase_unwrapped = np.unwrap(phase, axis=1)
    # Instantaneous frequency-like feature: derivative of unwrapped phase
    phase_diff = np.diff(phase_unwrapped, axis=1, prepend=phase_unwrapped[:, :1])
    phase_diff = phase_diff.astype(np.float32)
    phase_diff /= (np.abs(phase_diff).max() + 1e-10)

    return np.stack([mag, phase_diff], axis=0)


# --------------------------------------------------------------------------- #
# RF-specific data augmentation
# --------------------------------------------------------------------------- #

def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add complex additive white Gaussian noise at a target SNR (dB)."""
    rng = rng or np.random.default_rng()
    x = iq if np.iscomplexobj(iq) else iq[0] + 1j * iq[1]

    sig_power = np.mean(np.abs(x) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)) * np.sqrt(noise_power / 2)
    out = x + noise

    if np.iscomplexobj(iq):
        return out
    return np.stack([out.real, out.imag], axis=0)


def random_freq_shift(
    iq: np.ndarray, max_shift_hz: float, sample_rate: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Apply a random carrier frequency offset (CFO) — common real-world RF impairment."""
    rng = rng or np.random.default_rng()
    x = iq if np.iscomplexobj(iq) else iq[0] + 1j * iq[1]

    shift_hz = rng.uniform(-max_shift_hz, max_shift_hz)
    t = np.arange(len(x)) / sample_rate
    out = x * np.exp(1j * 2 * np.pi * shift_hz * t)

    if np.iscomplexobj(iq):
        return out
    return np.stack([out.real, out.imag], axis=0)


def random_phase_offset(iq: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Rotate the constellation by a random phase — simulates oscillator phase noise."""
    rng = rng or np.random.default_rng()
    x = iq if np.iscomplexobj(iq) else iq[0] + 1j * iq[1]
    theta = rng.uniform(0, 2 * np.pi)
    out = x * np.exp(1j * theta)
    if np.iscomplexobj(iq):
        return out
    return np.stack([out.real, out.imag], axis=0)


def random_time_shift(iq: np.ndarray, max_shift: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Circular time shift — simulates unknown burst start offset."""
    rng = rng or np.random.default_rng()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    axis = -1 if iq.ndim == 1 else 1
    return np.roll(iq, shift, axis=axis)


def spec_augment(
    spec: np.ndarray,
    freq_mask_width: int = 8,
    time_mask_width: int = 8,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    SpecAugment-style masking on an already-computed spectrogram.
    Expects shape (F, T) or (C, F, T).
    """
    rng = rng or np.random.default_rng()
    out = spec.copy()
    freq_axis = -2
    time_axis = -1
    n_freq = out.shape[freq_axis]
    n_time = out.shape[time_axis]

    for _ in range(n_freq_masks):
        w = rng.integers(0, freq_mask_width + 1)
        if w == 0 or w >= n_freq:
            continue
        f0 = rng.integers(0, n_freq - w)
        if out.ndim == 2:
            out[f0:f0 + w, :] = 0
        else:
            out[:, f0:f0 + w, :] = 0

    for _ in range(n_time_masks):
        w = rng.integers(0, time_mask_width + 1)
        if w == 0 or w >= n_time:
            continue
        t0 = rng.integers(0, n_time - w)
        if out.ndim == 2:
            out[:, t0:t0 + w] = 0
        else:
            out[:, :, t0:t0 + w] = 0

    return out


class RFAugmentPipeline:
    """
    Composable IQ-domain augmentation pipeline applied before spectrogram
    generation. Configure once, call per-sample inside the Dataset.
    """

    def __init__(
        self,
        snr_range_db: tuple[float, float] | None = (5.0, 25.0),
        max_freq_shift_hz: float = 0.0,
        sample_rate: float = 1.0,
        apply_phase_offset: bool = True,
        max_time_shift: int = 0,
        p: float = 0.5,
        seed: int | None = None,
    ):
        self.snr_range_db = snr_range_db
        self.max_freq_shift_hz = max_freq_shift_hz
        self.sample_rate = sample_rate
        self.apply_phase_offset = apply_phase_offset
        self.max_time_shift = max_time_shift
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        out = iq
        if self.snr_range_db is not None and self.rng.random() < self.p:
            snr = self.rng.uniform(*self.snr_range_db)
            out = add_awgn(out, snr, rng=self.rng)
        if self.max_freq_shift_hz > 0 and self.rng.random() < self.p:
            out = random_freq_shift(out, self.max_freq_shift_hz, self.sample_rate, rng=self.rng)
        if self.apply_phase_offset and self.rng.random() < self.p:
            out = random_phase_offset(out, rng=self.rng)
        if self.max_time_shift > 0 and self.rng.random() < self.p:
            out = random_time_shift(out, self.max_time_shift, rng=self.rng)
        return out
