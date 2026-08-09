"""SpectraNet inference API - a real served endpoint. Loads a trained
checkpoint and exposes GET /health, GET /classes, POST /predict over HTTP."""

from __future__ import annotations

import csv
import os
from typing import Dict, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from spectranet.data.preprocessing import normalize_iq, spectrogram
from spectranet.models.zoo import build_model

N_FFT = 128
HOP_LENGTH = 32

CHECKPOINT_PATH = os.environ.get("SPECTRANET_CHECKPOINT", "checkpoints/threat_classifier_resnet18_best.pt")

app = FastAPI(title="SpectraNet Inference API", description="RF signal / threat classification, served.", version="0.1.0")

_model = None
_class_names: Dict[int, str] = {}


class IQRequest(BaseModel):
    iq_real: List[float]
    iq_imag: List[float]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


def _load_class_names(data_root):
    mapping = {}
    path = os.path.join(data_root, "class_names.csv")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["label"])] = row["class_name"]
    return mapping


def _load_model():
    global _model, _class_names

    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}. Set SPECTRANET_CHECKPOINT.")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = ckpt["config"]

    model = build_model(cfg["model_name"], in_channels=cfg["in_channels"], num_classes=cfg["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model = model
    _class_names = _load_class_names(cfg["data_root"])


@app.on_event("startup")
def startup_event():
    _load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "checkpoint": CHECKPOINT_PATH}


@app.get("/classes")
def classes():
    return _class_names


@app.post("/predict", response_model=PredictionResponse)
def predict(request: IQRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.iq_real) != len(request.iq_imag):
        raise HTTPException(status_code=400, detail="iq_real and iq_imag must be the same length")
    if len(request.iq_real) < N_FFT:
        raise HTTPException(status_code=400, detail=f"Signal must have at least {N_FFT} samples")

    iq_real = np.array(request.iq_real, dtype=np.float32)
    iq_imag = np.array(request.iq_imag, dtype=np.float32)
    iq = iq_real + 1j * iq_imag

    iq = normalize_iq(iq, mode="unit_energy")
    spec = spectrogram(iq, n_fft=N_FFT, hop_length=HOP_LENGTH)
    tensor = torch.from_numpy(spec[np.newaxis, np.newaxis, ...].astype(np.float32))

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    probabilities = {_class_names.get(i, str(i)): float(probs[i]) for i in range(len(probs))}

    return PredictionResponse(
        predicted_class=_class_names.get(pred_idx, str(pred_idx)),
        confidence=float(probs[pred_idx]),
        probabilities=probabilities,
    )
