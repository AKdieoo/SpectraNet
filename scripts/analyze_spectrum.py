"""Spectral analysis using FFT - a real RF workflow step (quick single-shot
FFT magnitude spectrum per class) run before the deeper STFT/spectrogram
pipeline. Actually calls compute_fft() end to end on real dataset samples."""

from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import argparse
import csv
from pathlib import Path

import numpy as np

from spectranet.data.preprocessing import compute_fft, normalize_iq


def load_index_simple(root, index_file):
    rows = []
    with open(Path(root) / index_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], int(row["label"])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--index-file", type=str, default="train.csv")
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--out", type=str, default="fft_spectrum_analysis.csv")
    args = parser.parse_args()

    root = Path(args.data_root)
    samples = load_index_simple(args.data_root, args.index_file)

    by_label = {}
    for path, label in samples:
        by_label.setdefault(label, []).append(path)

    print(f"Loaded index with {len(samples)} samples across {len(by_label)} classes")
    print(f"Running FFT (n_fft={args.n_fft}) per class via compute_fft()...\n")

    results = []
    for label in sorted(by_label):
        paths = by_label[label]
        mag_sum = np.zeros(args.n_fft)
        n = 0
        for p in paths:
            iq = np.load(root / p)
            iq = normalize_iq(iq, mode="unit_energy")
            spectrum = compute_fft(iq, n_fft=args.n_fft, shift=True)
            mag_sum += np.abs(spectrum)
            n += 1
        mag_avg = mag_sum / n

        peak = mag_avg.max()
        occupied_bins = int(np.sum(mag_avg > 0.1 * peak))
        occupied_fraction = occupied_bins / args.n_fft

        print(f"Class {label}: {n} samples | peak FFT magnitude={peak:.3f} "
              f"| occupied bandwidth={occupied_fraction*100:.1f}% of spectrum")

        results.append({
            "label": label,
            "n_samples": n,
            "peak_fft_magnitude": peak,
            "occupied_bandwidth_fraction": occupied_fraction,
        })

    fieldnames = list(results[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved per-class FFT spectral analysis to {args.out}")


if __name__ == "__main__":
    main()
