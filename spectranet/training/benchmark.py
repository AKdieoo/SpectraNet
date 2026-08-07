"""
Benchmark trained architectures across accuracy, inference latency, and
model size.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from spectranet.data.dataset import RFIQDataset, RFSpectrogramDataset
from spectranet.models.zoo import build_model, count_parameters

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def measure_latency(model, input_shape, device, n_warmup=20, n_iters=100, batch_size=1):
    model.eval()
    dummy = torch.randn(batch_size, *input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        timings = []
        with torch.no_grad():
            for _ in range(n_iters):
                start = time.perf_counter()
                model(dummy)
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)

    timings = np.array(timings)
    return {
        "latency_mean_ms": float(timings.mean()),
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
        "throughput_samples_per_sec": float(batch_size * 1000 / timings.mean()),
    }


def measure_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total else float("nan")


def model_size_mb(model):
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
    finally:
        os.remove(tmp_path)
    return size_bytes / (1024 ** 2)


def benchmark_checkpoint(ckpt_path, model_name, data_root, val_index, input_shape, num_classes, device, use_raw_iq=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name, in_channels=input_shape[0], num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if use_raw_iq:
        val_ds = RFIQDataset(data_root, index_file=val_index)
    else:
        val_ds = RFSpectrogramDataset(data_root, index_file=val_index)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    accuracy = measure_accuracy(model, val_loader, device)
    latency = measure_latency(model, input_shape, device)
    size_mb = model_size_mb(model)
    params = count_parameters(model)

    return {
        "model": model_name,
        "checkpoint": ckpt_path,
        "accuracy": accuracy,
        "params_millions": params / 1e6,
        "size_mb": size_mb,
        **latency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--val-index", type=str, default="val.csv")
    parser.add_argument("--input-shape", type=str, default="1,128,128")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="benchmark_results.csv")
    parser.add_argument("--mlflow-experiment", type=str, default="spectranet-benchmark")
    parser.add_argument("--use-raw-iq", action="store_true")
    args = parser.parse_args()

    assert len(args.models) == len(args.checkpoints), "models and checkpoints must align 1:1"
    input_shape = tuple(int(v) for v in args.input_shape.split(","))

    results = []
    for model_name, ckpt_path in zip(args.models, args.checkpoints):
        print(f"Benchmarking {model_name} ...")
        result = benchmark_checkpoint(
            ckpt_path, model_name, args.data_root, args.val_index,
            input_shape, args.num_classes, args.device, use_raw_iq=args.use_raw_iq,
        )
        results.append(result)

    fieldnames = list(results[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    header = f"{'Model':<20}{'Acc':>8}{'Params(M)':>12}{'Size(MB)':>12}{'Lat p50(ms)':>14}{'Thrpt/s':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<20}{r['accuracy']:>8.4f}{r['params_millions']:>12.2f}"
            f"{r['size_mb']:>12.2f}{r['latency_p50_ms']:>14.2f}{r['throughput_samples_per_sec']:>10.1f}"
        )

    if _HAS_MLFLOW:
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name="architecture_comparison"):
            for r in results:
                with mlflow.start_run(run_name=r["model"], nested=True):
                    mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(args.out)

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
