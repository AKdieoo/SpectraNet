"""SpectraNet Live Command Center - Phase 1. Simulated live feed (no SDR
hardware) but every number is a REAL output from the real trained model via
the real API - no fake/scripted numbers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import requests
import streamlit as st

from generate_threat_dataset import make_benign, make_threat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spectranet.data.preprocessing import normalize_iq, spectrogram

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="SpectraNet Live Command Center", layout="wide")


def check_api_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        return resp.json()
    except requests.exceptions.RequestException:
        return None


def generate_and_classify():
    rng = np.random.default_rng()
    kind = rng.choice(["benign", "threat"])
    iq = make_benign(rng) if kind == "benign" else make_threat(rng)

    iq_norm = normalize_iq(iq, mode="unit_energy")
    spec = spectrogram(iq_norm, n_fft=128, hop_length=32)
    spectrum_bins = spec.mean(axis=1)

    payload = {"iq_real": iq.real.tolist(), "iq_imag": iq.imag.tolist()}
    start = time.perf_counter()
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
    latency_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    result = response.json()

    threat_score = round(result["probabilities"].get("threat", 0.0) * 100, 1)

    return {
        "timestamp": time.strftime("%H:%M:%S"),
        "ground_truth": kind,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "threat_score": threat_score,
        "latency_ms": round(latency_ms, 2),
        "spectrum": spectrum_bins,
        "correct": result["predicted_class"] == kind,
    }


if "history" not in st.session_state:
    st.session_state.history = []
if "auto_mode" not in st.session_state:
    st.session_state.auto_mode = False

st.title("SpectraNet Live Command Center")
st.caption(
    "Simulated live signal feed (no SDR hardware) - but every prediction, "
    "confidence, threat score, and latency below is a REAL output from the "
    "actual trained model, served over the actual API."
)

health = check_api_health()
if health is None:
    st.error(
        "Cannot reach the SpectraNet API at " + API_URL + ". "
        "Make sure `uvicorn spectranet.serve.api:app --port 8000` is running "
        "in another terminal."
    )
    st.stop()
else:
    st.success(f"Connected to SpectraNet API ? model loaded: {health['model_loaded']} "
               f"(checkpoint: {health['checkpoint']})")

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("Generate next signal"):
        st.session_state.history.append(generate_and_classify())
with col2:
    st.session_state.auto_mode = st.checkbox("Auto mode (every 2s)", value=st.session_state.auto_mode)

if st.session_state.history:
    latest = st.session_state.history[-1]

    st.subheader("Current Detection")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Class", latest["predicted_class"])
    m2.metric("Confidence", f"{latest['confidence']*100:.1f}%")
    m3.metric("Threat Score", f"{latest['threat_score']}/100")
    m4.metric("Latency", f"{latest['latency_ms']} ms")

    if latest["threat_score"] >= 70:
        st.error(f"HIGH THREAT SCORE ({latest['threat_score']}/100) - signal flagged for review.")
    elif latest["threat_score"] >= 40:
        st.warning(f"Moderate threat score ({latest['threat_score']}/100).")
    else:
        st.info(f"Low threat score ({latest['threat_score']}/100) - likely benign traffic.")

    st.subheader("Current Spectrum (real, from this signal)")
    spectrum_df = pd.DataFrame({"magnitude": latest["spectrum"]})
    st.bar_chart(spectrum_df)

    st.subheader("Detection Log")
    log_df = pd.DataFrame(st.session_state.history)[
        ["timestamp", "ground_truth", "predicted_class", "confidence", "threat_score", "latency_ms", "correct"]
    ]
    st.dataframe(log_df.iloc[::-1], use_container_width=True)

    n_correct = sum(h["correct"] for h in st.session_state.history)
    n_total = len(st.session_state.history)
    st.caption(f"Session accuracy: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")
else:
    st.info("Click 'Generate next signal' to see your first real detection.")

if st.session_state.auto_mode:
    time.sleep(2)
    st.session_state.history.append(generate_and_classify())
    st.rerun()
