"""Generate a small SYNTHETIC RF dataset so you can run the whole SpectraNet
pipeline end-to-end right now, without needing a real dataset yet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SEQ_LEN = 1024
SAMPLE_RATE = 1.0


def make_symbols(n_symbols, constellation, rng):
    idx = rng.integers(0, len(constellation), size=n_symbols)
    return constellation[idx]


def modulate(class_name, rng):
    if class_name == "bpsk":
        const = np.array([1 + 0j, -1 + 0j])
        symbols = make_symbols(SEQ_LEN // 8, const, rng)
    elif class_name == "qpsk":
        const = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        symbols = make_symbols(SEQ_LEN // 8, const, rng)
    elif class_name == "psk8":
        angles = np.arange(8) * (2 * np.pi / 8)
        const = np.exp(1j * angles)
        symbols = make_symbols(SEQ_LEN // 8, const, rng)
    elif class_name == "qam16":
        levels = np.array([-3, -1, 1, 3])
        const = np.array([i + 1j * q for i in levels for q in levels]) / np.sqrt(10)
        symbols = make_symbols(SEQ_LEN // 8, const, rng)
    elif class_name == "noise":
        iq = (rng.normal(size=SEQ_LEN) + 1j * rng.normal(size=SEQ_LEN)) * 0.5
        return iq.astype(np.complex64)
    else:
        raise ValueError(f"Unknown class: {class_name}")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--n-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    classes = ["bpsk", "qpsk", "psk8", "qam16", "noise"]
    rng = np.random.default_rng(args.seed)

    root = Path(args.out) / "samples"
    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(args.n_per_class):
            iq = modulate(class_name, rng)
            np.save(class_dir / f"sample_{i:04d}.npy", iq)
        print(f"Generated {args.n_per_class} samples for class '{class_name}'")

    print(f"\nDone. Synthetic dataset written to: {Path(args.out).resolve()}")
    print("Next: run scripts/build_index.py to create train.csv / val.csv")


if __name__ == "__main__":
    main()
