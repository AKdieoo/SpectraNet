"""Tests for spectranet/context/store.py (Phase 3 metadata store).

Uses an in-memory SQLite DB per test (conn_string=":memory:") so tests are
isolated and leave no files on disk.
"""

import pytest

from spectranet.context.models import Dataset, Experiment, Model, Prediction, Signal
from spectranet.context.store import ContextStore


@pytest.fixture
def store():
    return ContextStore(conn_string=":memory:")


def test_save_and_get_roundtrip(store):
    ds = Dataset.create(name="clean_synthetic_v1", class_names=["bpsk", "qpsk"])
    store.save(ds)

    fetched = store.get(Dataset, ds.id)
    assert fetched is not None
    assert fetched.id == ds.id
    assert fetched.name == "clean_synthetic_v1"
    assert fetched.class_names == ["bpsk", "qpsk"]


def test_get_missing_returns_none(store):
    assert store.get(Dataset, "ds-doesnotexist") is None


def test_save_is_upsert(store):
    ds = Dataset.create(name="v1")
    store.save(ds)

    ds.name = "v1-renamed"
    store.save(ds)

    fetched = store.get(Dataset, ds.id)
    assert fetched.name == "v1-renamed"
    # still only one row for this id
    assert len(store.list(Dataset)) == 1


def test_list_all_and_by_status(store):
    a = Dataset.create(name="a")
    b = Dataset.create(name="b")
    store.save(a)
    store.save(b)

    all_ds = store.list(Dataset)
    assert len(all_ds) == 2

    active = store.list(Dataset, status="active")
    assert len(active) == 2


def test_delete(store):
    ds = Dataset.create(name="to_delete")
    store.save(ds)
    assert store.delete(Dataset, ds.id) is True
    assert store.get(Dataset, ds.id) is None
    assert store.delete(Dataset, ds.id) is False  # already gone


def test_find_by_relationship_field(store):
    ds = Dataset.create(name="threat_v1")
    exp = Experiment.create(name="run1", dataset_id=ds.id, architecture="resnet18")
    model = Model.create(name="resnet18_v1.4", experiment_id=exp.id, architecture="resnet18")
    sig = Signal.create(name="SIG-1842", signal_type="sweep_jammer")

    p1 = Prediction.create(name="p1", signal_id=sig.id, model_id=model.id, label="threat", confidence=0.97)
    p2 = Prediction.create(name="p2", signal_id="sig-other", model_id=model.id, label="benign", confidence=0.9)
    store.save(p1)
    store.save(p2)

    results = store.find_by(Prediction, signal_id=sig.id)
    assert len(results) == 1
    assert results[0].id == p1.id


def test_full_chain_persists_and_reconstructs(store):
    """End-to-end: build a full Dataset -> Experiment -> Model -> Signal ->
    Prediction chain, save everything, and verify it can be reconstructed
    purely from the store (this is what lineage/Phase 5 will build on)."""
    ds = Dataset.create(name="threat_v1", class_names=["benign", "threat"])
    exp = Experiment.create(name="resnet18_run1", dataset_id=ds.id, architecture="resnet18", val_accuracy=1.0)
    model = Model.create(name="resnet18_v1.4", experiment_id=exp.id, architecture="resnet18", validation_accuracy=1.0)
    sig = Signal.create(name="SIG-1842", signal_type="sweep_jammer", snr_db=12.0)
    pred = Prediction.create(name="pred-1842", signal_id=sig.id, model_id=model.id, label="threat", confidence=0.9761)

    for entity in (ds, exp, model, sig, pred):
        store.save(entity)

    fetched_pred = store.get(Prediction, pred.id)
    fetched_model = store.get(Model, fetched_pred.model_id)
    fetched_exp = store.get(Experiment, fetched_model.experiment_id)
    fetched_ds = store.get(Dataset, fetched_exp.dataset_id)

    assert fetched_ds.name == "threat_v1"
    assert fetched_exp.architecture == "resnet18"
    assert fetched_model.validation_accuracy == pytest.approx(1.0)
    assert fetched_pred.confidence == pytest.approx(0.9761)
