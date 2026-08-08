"""Precompute spectrograms from an existing raw-IQ dataset, so
RFSpectrogramDataset (as opposed to RFIQDataset's on-the-fly path) can be
exercised for real with real data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectranet.data.preprocessing import normalize_iq, spectrogram


def load_index_simple(root, index_file):
    rows = []
    with open(Path(root) / index_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], int(row["label"])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    args = parser.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    class_names = {}
    with open(in_root / "class_names.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names[int(row["label"])] = row["class_name"]

    total = 0
    for index_file in ["train.csv", "val.csv"]:
        rows = load_index_simple(str(in_root), index_file)
        for rel_path, label in rows:
            iq = np.load(in_root / rel_path)
            iq = normalize_iq(iq, mode="unit_energy")
            spec = spectrogram(iq, n_fft=args.n_fft, hop_length=args.hop_length)

            class_name = class_names[label]
            out_dir = out_root / "samples" / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            sample_name = Path(rel_path).name
            np.save(out_dir / sample_name, spec.astype(np.float32))
            total += 1

    print(f"Precomputed {total} spectrograms -> {out_root.resolve()}")
    print("Next: run scripts/build_index.py --data-root <out-root> to create train.csv/val.csv")


if __name__ == "__main__":
    main()
