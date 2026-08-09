"""Test client for the SpectraNet API. Generates a real signal and sends it
over HTTP to a running server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import requests

from generate_threat_dataset import make_benign, make_threat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--kind", choices=["benign", "threat"], default="threat")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    iq = make_benign(rng) if args.kind == "benign" else make_threat(rng)

    print(f"Checking server health at {args.url}/health ...")
    health = requests.get(f"{args.url}/health").json()
    print(f"  {health}")

    print(f"\nGenerated a real '{args.kind}' signal ({len(iq)} samples), sending to /predict ...")
    payload = {"iq_real": iq.real.tolist(), "iq_imag": iq.imag.tolist()}
    response = requests.post(f"{args.url}/predict", json=payload)
    response.raise_for_status()
    result = response.json()

    print(f"\nPredicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("All class probabilities:")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls}: {prob:.4f}")

    correct = "YES" if result["predicted_class"] == args.kind else "NO"
    print(f"\nGround truth was '{args.kind}' -> correct: {correct}")


if __name__ == "__main__":
    main()
