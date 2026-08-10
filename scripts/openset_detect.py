"""Open-set / unknown-signal detection - Maximum Softmax Probability baseline
(Hendrycks & Gimpel, 2017). Tested honestly: jammer signals genuinely never
seen by the modulation classifier are used as real out-of-distribution data."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from generate_threat_dataset import make_threat
from spectranet.data.preprocessing import normalize_iq, spectrogram
from spectranet.models.zoo import build_model


def load_class_names(data_root):
    mapping = {}
    with open(Path(data_root) / "class_names.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["label"])] = row["class_name"]
    return mapping


def load_index_simple(root, index_file):
    rows = []
    with open(Path(root) / index_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["path"], int(row["label"])))
    return rows


def get_max_softmax_prob(model, spec):
    tensor = torch.from_numpy(spec[np.newaxis, np.newaxis, ...].astype(np.float32))
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    max_prob, pred_idx = torch.max(probs, dim=0)
    return float(max_prob), int(pred_idx)


def calibrate_threshold(model, data_root, index_file, percentile=5.0, n_samples=200):
    rows = load_index_simple(data_root, index_file)[:n_samples]
    max_probs = []
    for rel_path, _ in rows:
        iq = np.load(Path(data_root) / rel_path)
        iq_norm = normalize_iq(iq, mode="unit_energy")
        spec = spectrogram(iq_norm, n_fft=128, hop_length=32)
        max_prob, _ = get_max_softmax_prob(model, spec)
        max_probs.append(max_prob)
    threshold = float(np.percentile(max_probs, percentile))
    return threshold, max_probs


def evaluate_ood_detection(model, threshold, n_ood_samples=100):
    rng = np.random.default_rng(999)
    flagged_unknown = 0
    max_probs = []

    for _ in range(n_ood_samples):
        iq = make_threat(rng)
        iq_norm = normalize_iq(iq, mode="unit_energy")
        spec = spectrogram(iq_norm, n_fft=128, hop_length=32)
        max_prob, pred_idx = get_max_softmax_prob(model, spec)
        max_probs.append(max_prob)
        if max_prob < threshold:
            flagged_unknown += 1

    detection_rate = flagged_unknown / n_ood_samples
    return {
        "n_ood_samples": n_ood_samples,
        "flagged_unknown": flagged_unknown,
        "detection_rate": detection_rate,
        "mean_max_prob_on_ood": float(np.mean(max_probs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--percentile", type=float, default=5.0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg["model_name"], in_channels=cfg["in_channels"], num_classes=cfg["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_names = load_class_names(args.data_root)

    print("Step 1: Calibrating threshold on REAL in-distribution validation data...")
    threshold, id_max_probs = calibrate_threshold(model, args.data_root, "val.csv", args.percentile)
    print(f"  Threshold: {threshold:.4f} (calibrated so {args.percentile}% of KNOWN signals would be falsely flagged unknown)")
    print(f"  Mean max-softmax-prob on known signals: {np.mean(id_max_probs):.4f}\n")

    print("Step 2: Testing on REAL out-of-distribution signals (jammers - never seen by this model)...")
    ood_results = evaluate_ood_detection(model, threshold, n_ood_samples=100)
    print(f"  Out of {ood_results['n_ood_samples']} genuinely unseen jammer signals:")
    print(f"  Flagged as UNKNOWN: {ood_results['flagged_unknown']} ({ood_results['detection_rate']*100:.1f}%)")
    print(f"  Mean max-softmax-prob on OOD signals: {ood_results['mean_max_prob_on_ood']:.4f}")
    print(f"  (compare to {np.mean(id_max_probs):.4f} on known signals - lower = model is less confident on things it's never seen, as expected)")


if __name__ == "__main__":
    main()
