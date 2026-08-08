"""Generate a genuine binary RF THREAT CLASSIFICATION dataset: benign
(legitimate BPSK/QPSK/8PSK/QAM16 comms) vs threat (real jamming types:
sweep, barrage, pulsed). Power-matched across classes so the model must
learn actual waveform shape, not just amplitude/loudness as a shortcut."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SEQ_LEN = 1024
SAMPLE_RATE = 1.0


def make_symbols(n_symbols, constellation, rng):
    idx = rng.integers(0, len(constellation), size=n_symbols)
    return constellation[idx]


def make_benign(rng):
    kind = rng.choice(["bpsk", "qpsk", "psk8", "qam16"])

    if kind == "bpsk":
        const = np.array([1 + 0j, -1 + 0j])
        symbols = make_symbols(SEQ_LEN // 8, const, rng)
    elif kind == "qpsk":
        const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        symbols = make_symbols(SEQ_LEN // 8, const, rng)
    elif kind == "psk8":
        angles = np.arange(8) * (2 * np.pi / 8)
        const = np.exp(1j * angles)
        symbols = make_symbols(SEQ_LEN // 8, const, rng)
    else:
        levels = np.array([-3, -1, 1, 3])
        const = np.array([i + 1j * q for i in levels for q in levels]) / np.sqrt(10)
        symbols = make_symbols(SEQ_LEN // 8, const, rng)

    samples_per_symbol = SEQ_LEN // len(symbols)
    iq = np.repeat(symbols, samples_per_symbol)
    iq = iq[:SEQ_LEN]
    if len(iq) < SEQ_LEN:
        iq = np.pad(iq, (0, SEQ_LEN - len(iq)))

    t = np.arange(SEQ_LEN) / SAMPLE_RATE
    cfo = rng.uniform(-0.01, 0.01)
    iq = iq * np.exp(1j * 2 * np.pi * cfo * t)
    noise = (rng.normal(size=SEQ_LEN) + 1j * rng.normal(size=SEQ_LEN)) * 0.05
    iq = iq + noise

    return iq.astype(np.complex64)


def make_sweep_jammer(rng):
    t = np.arange(SEQ_LEN) / SAMPLE_RATE
    f_start = rng.uniform(-0.45, -0.1)
    f_end = rng.uniform(0.1, 0.45)
    inst_freq = f_start + (f_end - f_start) * (t / SEQ_LEN)
    phase = 2 * np.pi * np.cumsum(inst_freq)
    amplitude = rng.uniform(0.85, 1.15)
    iq = amplitude * np.exp(1j * phase)
    noise = (rng.normal(size=SEQ_LEN) + 1j * rng.normal(size=SEQ_LEN)) * 0.1
    return (iq + noise).astype(np.complex64)


def make_barrage_jammer(rng):
    target_power = rng.uniform(0.85, 1.15)
    amplitude = np.sqrt(target_power / 2)
    iq = (rng.normal(size=SEQ_LEN) + 1j * rng.normal(size=SEQ_LEN)) * amplitude
    return iq.astype(np.complex64)


def make_pulsed_jammer(rng):
    iq = np.zeros(SEQ_LEN, dtype=np.complex64)
    pulse_width = rng.integers(30, 100)
    gap_width = rng.integers(20, 80)
    duty_fraction = pulse_width / (pulse_width + gap_width)
    target_power = rng.uniform(0.85, 1.15)
    on_amplitude = np.sqrt(target_power / max(duty_fraction, 0.05))

    pos = 0
    carrier_freq = rng.uniform(-0.3, 0.3)
    t_full = np.arange(SEQ_LEN) / SAMPLE_RATE
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t_full)

    while pos < SEQ_LEN:
        end = min(pos + pulse_width, SEQ_LEN)
        iq[pos:end] = on_amplitude * carrier[pos:end]
        pos = end + gap_width

    noise = (rng.normal(size=SEQ_LEN) + 1j * rng.normal(size=SEQ_LEN)) * 0.1
    return (iq + noise).astype(np.complex64)


def make_threat(rng):
    kind = rng.choice(["sweep", "barrage", "pulsed"])
    if kind == "sweep":
        return make_sweep_jammer(rng)
    elif kind == "barrage":
        return make_barrage_jammer(rng)
    else:
        return make_pulsed_jammer(rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/threat")
    parser.add_argument("--n-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out) / "samples"

    for class_name, generator in [("benign", make_benign), ("threat", make_threat)]:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n_per_class):
            iq = generator(rng)
            np.save(class_dir / f"sample_{i:04d}.npy", iq)
        print(f"Generated {args.n_per_class} '{class_name}' samples")

    print(f"\nDone. Threat-classification dataset written to: {Path(args.out).resolve()}")
    print("Classes: benign (BPSK/QPSK/8PSK/QAM16 comms), threat (sweep/barrage/pulsed jamming)")
    print("Next: run scripts/build_index.py to create train.csv / val.csv")


if __name__ == "__main__":
    main()
