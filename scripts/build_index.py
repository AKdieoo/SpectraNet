"""
Build train.csv / val.csv index files for RFIQDataset / RFSpectrogramDataset
from a directory tree organized as:

    data_root/
      samples/
        <class_name>/
          sample_0001.npy
          sample_0002.npy
          ...

Each .npy file should contain either a complex 1D array or a (2, N) real
array of I/Q samples (or, if --precomputed, an already-computed
spectrogram of shape (C, F, T)).

Usage
-----
    python scripts/build_index.py --data-root data/processed --val-fraction 0.15
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--samples-subdir", type=str, default="samples")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.data_root)
    samples_dir = root / args.samples_subdir
    class_dirs = sorted(p for p in samples_dir.iterdir() if p.is_dir())

    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found under {samples_dir}")

    class_names = [d.name for d in class_dirs]
    label_map = {name: i for i, name in enumerate(class_names)}

    rows = []
    for class_dir in class_dirs:
        label = label_map[class_dir.name]
        for f in sorted(class_dir.glob("*.npy")):
            rel_path = f.relative_to(root).as_posix()
            rows.append((rel_path, label))

    random.Random(args.seed).shuffle(rows)
    n_val = int(len(rows) * args.val_fraction)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    for name, subset in [("train.csv", train_rows), ("val.csv", val_rows)]:
        with open(root / name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows(subset)
        print(f"Wrote {len(subset)} rows to {root / name}")

    with open(root / "class_names.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "class_name"])
        for name, idx in label_map.items():
            writer.writerow([idx, name])

    print(f"Classes ({len(class_names)}): {class_names}")


if __name__ == "__main__":
    main()
