"""
Custom training loop for SpectraNet models with dual experiment tracking
(Weights & Biases + MLflow) side by side.

Both trackers are optional at runtime (guarded imports) so the training
loop still runs in environments without network access to either service;
set `use_wandb=False` / `use_mlflow=False` in the config to disable
explicitly.

Usage
-----
    python -m spectranet.training.train --config configs/resnet18.yaml
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from spectranet.data.dataset import RFIQDataset, RFSpectrogramDataset
from spectranet.data.preprocessing import RFAugmentPipeline
from spectranet.models.zoo import build_model, count_parameters

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


@dataclass
class TrainConfig:
    model_name: str = "resnet18"
    data_root: str = "data/processed"
    train_index: str = "train.csv"
    val_index: str = "val.csv"
    in_channels: int = 1
    num_classes: int = 10
    pretrained: bool = True
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine" | "step" | "none"
    label_smoothing: float = 0.05
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb: bool = True
    use_mlflow: bool = True
    wandb_project: str = "spectranet"
    mlflow_experiment: str = "spectranet"
    checkpoint_dir: str = "checkpoints"
    early_stopping_patience: int = 8
    use_raw_iq: bool = False  # True -> RFIQDataset (on-the-fly STFT), False -> precomputed
    augment: bool = True
    seed: int = 42
    run_name: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, int]:
    if cfg.use_raw_iq:
        augment = RFAugmentPipeline(seed=cfg.seed) if cfg.augment else None
        train_ds = RFIQDataset(cfg.data_root, index_file=cfg.train_index, augment=augment)
        val_ds = RFIQDataset(cfg.data_root, index_file=cfg.val_index, augment=None)
    else:
        train_ds = RFSpectrogramDataset(cfg.data_root, index_file=cfg.train_index)
        val_ds = RFSpectrogramDataset(cfg.data_root, index_file=cfg.val_index)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds.num_classes


def build_optimizer_and_scheduler(model: nn.Module, cfg: TrainConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    else:
        scheduler = None

    return optimizer, scheduler


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train(cfg: TrainConfig) -> str:
    torch.manual_seed(cfg.seed)
    run_name = cfg.run_name or f"{cfg.model_name}_{int(time.time())}"

    train_loader, val_loader, num_classes = build_dataloaders(cfg)
    num_classes = cfg.num_classes or num_classes

    model = build_model(
        cfg.model_name, in_channels=cfg.in_channels,
        num_classes=num_classes, pretrained=cfg.pretrained,
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)

    use_wandb = cfg.use_wandb and _HAS_WANDB
    use_mlflow = cfg.use_mlflow and _HAS_MLFLOW

    if use_wandb:
        wandb.init(project=cfg.wandb_project, name=run_name, config=cfg.__dict__)
    if use_mlflow:
        mlflow.set_experiment(cfg.mlflow_experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: v for k, v in cfg.__dict__.items() if not isinstance(v, dict)})
        mlflow.log_param("num_parameters", count_parameters(model))

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, cfg.device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, cfg.device, optimizer=None)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        log = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/accuracy": train_metrics["accuracy"],
            "val/loss": val_metrics["loss"],
            "val/accuracy": val_metrics["accuracy"],
            "lr": current_lr,
            "epoch_time_sec": epoch_time,
        }
        print(
            f"[{cfg.model_name}] epoch {epoch:03d}/{cfg.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"({epoch_time:.1f}s)"
        )

        if use_wandb:
            wandb.log(log)
        if use_mlflow:
            mlflow.log_metrics({k: v for k, v in log.items() if k != "epoch"}, step=epoch)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            epochs_without_improvement = 0
            best_path = ckpt_dir / f"{run_name}_best.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg.__dict__, "val_accuracy": best_val_acc},
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {cfg.early_stopping_patience} epochs)")
                break

    final_path = ckpt_dir / f"{run_name}_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, final_path)

    if use_mlflow:
        mlflow.log_artifact(str(best_path))
        mlflow.end_run()
    if use_wandb:
        wandb.log({"best_val_accuracy": best_val_acc})
        wandb.finish()

    print(f"Best val accuracy: {best_val_acc:.4f} | checkpoint: {best_path}")
    return str(best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
