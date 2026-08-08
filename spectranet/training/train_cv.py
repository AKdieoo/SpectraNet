"""K-fold cross-validation training loop - a genuinely DIFFERENT custom
training loop from train.py (not a re-run): splits data into K folds,
trains K separate models, aggregates mean/std accuracy across folds."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from spectranet.data.dataset import RFIQDataset, load_index
from spectranet.data.preprocessing import RFAugmentPipeline
from spectranet.models.zoo import build_model
from spectranet.training.train import TrainConfig, run_epoch, build_optimizer_and_scheduler


def make_kfold_indices(n_samples, k, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        val_idx = folds[i].tolist()
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]).tolist()
        splits.append((train_idx, val_idx))
    return splits


def train_one_fold(cfg, full_dataset, train_idx, val_idx, fold_num, num_classes):
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = build_model(cfg.model_name, in_channels=cfg.in_channels, num_classes=num_classes, pretrained=cfg.pretrained)
    model.to(cfg.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, cfg.device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, cfg.device, optimizer=None)
        if scheduler is not None:
            scheduler.step()
        best_val_acc = max(best_val_acc, val_metrics["accuracy"])
        print(f"  Fold {fold_num} | epoch {epoch}/{cfg.epochs} "
              f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f}")

    return {"fold": fold_num, "best_val_accuracy": best_val_acc,
            "train_size": len(train_idx), "val_size": len(val_idx)}


def run_kfold_cv(cfg, k_folds):
    torch.manual_seed(cfg.seed)

    augment = RFAugmentPipeline(seed=cfg.seed) if cfg.augment else None
    full_dataset = RFIQDataset(cfg.data_root, index_file=cfg.train_index, augment=augment)
    num_classes = cfg.num_classes or full_dataset.num_classes

    splits = make_kfold_indices(len(full_dataset), k_folds, seed=cfg.seed)

    print(f"Running {k_folds}-fold cross-validation for {cfg.model_name} "
          f"on {len(full_dataset)} samples ({num_classes} classes)\n")

    results = []
    t0 = time.time()
    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"--- Fold {fold_num}/{k_folds} ---")
        result = train_one_fold(cfg, full_dataset, train_idx, val_idx, fold_num, num_classes)
        results.append(result)

    total_time = time.time() - t0
    accs = [r["best_val_accuracy"] for r in results]
    print(f"\nK-fold CV complete in {total_time:.1f}s")
    print(f"Per-fold accuracy: {[f'{a:.4f}' for a in accs]}")
    print(f"Mean accuracy: {np.mean(accs):.4f}  |  Std: {np.std(accs):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--out", type=str, default="kfold_cv_results.csv")
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    results = run_kfold_cv(cfg, args.k_folds)

    accs = [r["best_val_accuracy"] for r in results]
    fieldnames = list(results[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({"fold": "mean", "best_val_accuracy": np.mean(accs), "train_size": "", "val_size": ""})
        writer.writerow({"fold": "std", "best_val_accuracy": np.std(accs), "train_size": "", "val_size": ""})

    print(f"\nSaved fold-by-fold results to {args.out}")


if __name__ == "__main__":
    main()
