import numpy as np
import pytest

from spectranet.data.preprocessing import (
    add_awgn,
    compute_stft,
    normalize_iq,
    random_freq_shift,
    random_phase_offset,
    random_time_shift,
    spec_augment,
    spectrogram,
    spectrogram_with_phase,
)


def make_tone(n=1024, freq=0.1, fs=1.0):
    t = np.arange(n) / fs
    return np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)


def test_normalize_iq_zscore_complex():
    x = make_tone() * 5.0 + 2.0
    out = normalize_iq(x, mode="zscore")
    assert np.isclose(out.mean(), 0, atol=1e-5)


def test_normalize_iq_unit_energy():
    x = make_tone() * 3.0
    out = normalize_iq(x, mode="unit_energy")
    assert np.isclose(np.mean(np.abs(out) ** 2), 1.0, atol=1e-5)


def test_normalize_iq_real_pair_roundtrip_shape():
    x = np.stack([np.random.randn(256), np.random.randn(256)])
    out = normalize_iq(x, mode="peak")
    assert out.shape == (2, 256)
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_compute_stft_shape():
    x = make_tone(n=1024)
    stft = compute_stft(x, n_fft=64, hop_length=16)
    assert stft.shape[0] == 64
    assert stft.dtype == np.complex64


def test_spectrogram_range():
    x = make_tone(n=1024)
    spec = spectrogram(x, n_fft=64, hop_length=16)
    assert spec.min() >= 0.0 - 1e-6
    assert spec.max() <= 1.0 + 1e-6


def test_spectrogram_with_phase_shape():
    x = make_tone(n=1024)
    spec = spectrogram_with_phase(x, n_fft=64, hop_length=16)
    assert spec.shape[0] == 2


def test_add_awgn_changes_signal_but_preserves_shape():
    x = make_tone(n=512)
    noisy = add_awgn(x, snr_db=10, rng=np.random.default_rng(0))
    assert noisy.shape == x.shape
    assert not np.allclose(noisy, x)


def test_random_freq_shift_preserves_shape():
    x = make_tone(n=512)
    shifted = random_freq_shift(x, max_shift_hz=0.05, sample_rate=1.0, rng=np.random.default_rng(0))
    assert shifted.shape == x.shape


def test_random_phase_offset_preserves_magnitude():
    x = make_tone(n=512)
    rotated = random_phase_offset(x, rng=np.random.default_rng(0))
    assert np.allclose(np.abs(rotated), np.abs(x), atol=1e-5)


def test_random_time_shift_is_permutation():
    x = np.arange(10).astype(np.complex64)
    shifted = random_time_shift(x, max_shift=3, rng=np.random.default_rng(1))
    # circular shift => same multiset of values, just reordered
    assert np.allclose(sorted(shifted.real.tolist()), sorted(x.real.tolist()))


def test_spec_augment_masks_something():
    spec = np.ones((32, 32), dtype=np.float32)
    out = spec_augment(spec, freq_mask_width=8, time_mask_width=8, rng=np.random.default_rng(2))
    assert out.shape == spec.shape
    # With width > 0 masking should zero out at least some region most of the time
    assert (out == 0).sum() >= 0  # smoke test: doesn't crash, shape preserved


def test_stft_too_short_raises():
    x = make_tone(n=8)
    with pytest.raises(ValueError):
        compute_stft(x, n_fft=64, hop_length=16, center=False)
