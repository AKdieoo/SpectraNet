"""Tests for spectranet/context/models.py (Phase 2 domain model).

Pure unit tests — no database, no ML, no network. Verifies entity
construction, ID generation, defaults, and that relationships (foreign-key
style references) hold together correctly.
"""

import pytest

from spectranet.context.models import (
    Dataset,
    Detection,
    Document,
    Evaluation,
    EntityStatus,
    Experiment,
    Explanation,
    Model,
    Prediction,
    Signal,
    TrustLevel,
    TrustSignal,
)


def test_dataset_create_generates_id_and_defaults():
    ds = Dataset.create(name="clean_synthetic_v1", class_names=["bpsk", "qpsk"], sample_count=1000)
    assert ds.id.startswith("ds-")
    assert ds.status == EntityStatus.ACTIVE
    assert ds.class_names == ["bpsk", "qpsk"]
    assert ds.created_at is not None


def test_experiment_references_dataset():
    ds = Dataset.create(name="clean_synthetic_v1")
    exp = Experiment.create(name="resnet18_run1", dataset_id=ds.id, architecture="resnet18", val_accuracy=0.627)
    assert exp.id.startswith("exp-")
    assert exp.dataset_id == ds.id
    assert exp.val_accuracy == 0.627


def test_model_references_experiment():
    ds = Dataset.create(name="clean_synthetic_v1")
    exp = Experiment.create(name="resnet18_run1", dataset_id=ds.id, architecture="resnet18")
    model = Model.create(
        name="resnet18_v1.4", experiment_id=exp.id, architecture="resnet18",
        validation_accuracy=0.627, known_limitations=["low-SNR accuracy drops to 57.8%"],
    )
    assert model.id.startswith("model-")
    assert model.experiment_id == exp.id
    assert "low-SNR accuracy drops to 57.8%" in model.known_limitations


def test_signal_and_detection_chain():
    sig = Signal.create(name="SIG-1842", signal_type="sweep_jammer", frequency=915e6, snr_db=12.0)
    det = Detection.create(name="detection-of-SIG-1842", signal_id=sig.id)
    assert det.signal_id == sig.id
    assert sig.id.startswith("sig-")
    assert det.id.startswith("det-")


def test_prediction_ties_signal_and_model():
    sig = Signal.create(name="SIG-1842", signal_type="sweep_jammer")
    ds = Dataset.create(name="threat_v1")
    exp = Experiment.create(name="run1", dataset_id=ds.id, architecture="resnet18")
    model = Model.create(name="resnet18_v1.4", experiment_id=exp.id, architecture="resnet18")

    pred = Prediction.create(
        name="pred-SIG-1842", signal_id=sig.id, model_id=model.id,
        label="threat", confidence=0.9761, probabilities={"threat": 0.9761, "benign": 0.0239},
    )
    assert pred.signal_id == sig.id
    assert pred.model_id == model.id
    assert pred.confidence == pytest.approx(0.9761)
    assert pred.probabilities["threat"] == pytest.approx(0.9761)


def test_explanation_references_prediction():
    expl = Explanation.create(
        name="gradcam-pred-1", prediction_id="pred-abc123", method="gradcam",
        summary="heatmap tracks the diagonal chirp trajectory",
    )
    assert expl.prediction_id == "pred-abc123"
    assert expl.method == "gradcam"


def test_trust_signal_defaults_to_medium():
    trust = TrustSignal.create(name="trust-pred-1", prediction_id="pred-abc123", confidence=0.9761, ood_score=0.08)
    assert trust.trust_level == TrustLevel.MEDIUM  # explicit decision logic comes in Phase 6
    assert trust.confidence == pytest.approx(0.9761)


def test_document_and_evaluation_construct_independently():
    doc = Document.create(name="rf-jamming-reference", topic="jamming", authority="internal")
    assert doc.id.startswith("doc-")

    ev = Evaluation.create(name="agent-eval-run1", target_type="agent", target_id="agent-xyz", metrics={"groundedness": 0.8})
    assert ev.target_type == "agent"
    assert ev.metrics["groundedness"] == 0.8


def test_entity_ids_are_unique():
    ids = {Dataset.create(name="x").id for _ in range(50)}
    assert len(ids) == 50
