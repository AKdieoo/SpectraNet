"""
Custom PyTorch Dataset(s) for RF signal classification.

Two datasets are provided:
  RFIQDataset          - loads raw IQ, applies augmentation + spectrogram
                          generation on the fly (best for research/augmentation
                          experimentation).
  RFSpectrogramDataset - loads pre-computed spectrograms directly (best for
                          fast training once preprocessing is finalized).

Expected on-disk layout (override `load_index` for your own dataset format,
e.g. RadioML2018.01A HDF5, GNU Radio captures, SigMF, etc.):

    root/
      index.csv            # columns: path,label,snr(optional)
      samples/*.npy         # each .npy is a raw complex IQ vector or (2,N) array

This mirrors common RF datasets (e.g. RadioML-style) closely enough that
swapping in a real dataset is mostly a matter of changing `load_index`.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from spectranet.data.preprocessing import (
    RFAugmentPipeline,
    normalize_iq,
    spectrogram,
    spectrogram_with_phase,
)


@dataclass
class RFSample:
    path: str
    label: int
    snr: Optional[float] = None


def load_index(root: str, index_file: str = "index.csv") -> list[RFSample]:
    """Read a simple CSV index: path,label[,snr]. Override for custom formats."""
    samples = []
    with open(os.path.join(root, index_file), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                RFSample(
                    path=row["path"],
                    label=int(row["label"]),
                    snr=float(row["snr"]) if row.get("snr") not in (None, "") else None,
                )
            )
    return samples


class RFIQDataset(Dataset):
    """
    Loads raw IQ samples, applies optional augmentation, converts to a
    spectrogram tensor on the fly.
    """

    def __init__(
        self,
        root: str,
        index_file: str = "index.csv",
        n_fft: int = 128,
        hop_length: int = 32,
        use_phase_channel: bool = False,
        normalize_mode: str = "unit_energy",
        augment: Optional[RFAugmentPipeline] = None,
        class_names: Optional[list[str]] = None,
        loader: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.root = root
        self.samples = load_index(root, index_file)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_phase_channel = use_phase_channel
        self.normalize_mode = normalize_mode
        self.augment = augment
        self.class_names = class_names
        self.loader = loader or self._default_loader

        if not self.samples:
            raise RuntimeError(f"No samples found under {root}/{index_file}")

    def _default_loader(self, path: str) -> np.ndarray:
        full = os.path.join(self.root, path)
        arr = np.load(full)
        return arr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        iq = self.loader(sample.path)
        iq = normalize_iq(iq, mode=self.normalize_mode)

        if self.augment is not None:
            iq = self.augment(iq)

        if self.use_phase_channel:
            spec = spectrogram_with_phase(iq, n_fft=self.n_fft, hop_length=self.hop_length)
        else:
            spec = spectrogram(iq, n_fft=self.n_fft, hop_length=self.hop_length)
            spec = spec[np.newaxis, ...]  # -> (1, F, T)

        tensor = torch.from_numpy(spec.astype(np.float32))
        return tensor, sample.label

    @property
    def num_classes(self) -> int:
        if self.class_names:
            return len(self.class_names)
        return len({s.label for s in self.samples})


class RFSpectrogramDataset(Dataset):
    """
    Loads pre-computed spectrogram tensors (.npy, shape (C, F, T)) directly.
    Use this once you've finalized preprocessing and want maximum training
    throughput (skips STFT computation every epoch).
    """

    def __init__(
        self,
        root: str,
        index_file: str = "index.csv",
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.root = root
        self.samples = load_index(root, index_file)
        self.transform = transform
        self.class_names = class_names

        if not self.samples:
            raise RuntimeError(f"No samples found under {root}/{index_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        spec = np.load(os.path.join(self.root, sample.path))
        if spec.ndim == 2:
            spec = spec[np.newaxis, ...]
        if self.transform is not None:
            spec = self.transform(spec)
        return torch.from_numpy(spec.astype(np.float32)), sample.label

    @property
    def num_classes(self) -> int:
        if self.class_names:
            return len(self.class_names)
        return len({s.label for s in self.samples})
