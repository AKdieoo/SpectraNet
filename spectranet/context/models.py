"""Domain / context model for the SpectraNet AI-native platform layer.

These entities give raw ML outputs meaning: they capture *what a signal is*,
*where a prediction came from*, *how confident we should be*, and *how it
all connects*. This module defines the entities and their relationships
only — no persistence, no ML calls, no LLM calls. Persistence lives in
`spectranet/context/store.py` (Phase 3).

Design notes
------------
- Every entity has: id, name/description, version/source where relevant,
  status, created_at, updated_at — matching the shared record shape called
  out in the build plan.
- Relationships are expressed as foreign-key-style ID references (e.g.
  Prediction.signal_id) rather than embedded objects, so entities can be
  persisted and queried independently. Lineage (Phase 5) walks these
  references to reconstruct the full chain.
- This module intentionally does NOT import anything from
  spectranet/data, spectranet/models, or spectranet/training. The context
  layer describes and wraps the existing ML platform; it never reaches
  into it directly. Only the future MCP tool layer (Phase 9) is allowed
  to call existing ML code, and only through the existing served API.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EntityStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    VALIDATED = "validated"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class TrustLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ABSTAIN = "abstain"


class BaseEntity(BaseModel):
    """Shared shape for every context entity."""

    id: str
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    source: Optional[str] = None
    status: EntityStatus = EntityStatus.ACTIVE
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Dataset(BaseEntity):
    """A named, versioned collection of samples used for training/eval.

    Maps to existing SpectraNet artifacts like data/processed/samples/ or
    the generated threat dataset — this entity just gives that folder a
    stable identity other entities can reference.
    """

    class_names: list[str] = Field(default_factory=list)
    sample_count: Optional[int] = None
    val_fraction: Optional[float] = None
    snr_profile: Optional[str] = None  # e.g. "clean" | "low_snr"

    @classmethod
    def create(cls, name: str, **kw) -> "Dataset":
        return cls(id=_new_id("ds"), name=name, **kw)


class Experiment(BaseEntity):
    """A single training run: config + resulting metrics.

    Wraps one run of spectranet/training/train.py or train_cv.py without
    re-implementing training logic here.
    """

    dataset_id: str
    architecture: str  # custom_cnn | resnet18 | vgg16 | efficientnet_b0 | mobilenet_v3_small
    config_path: Optional[str] = None
    val_accuracy: Optional[float] = None
    metrics: dict = Field(default_factory=dict)

    @classmethod
    def create(cls, name: str, dataset_id: str, architecture: str, **kw) -> "Experiment":
        return cls(id=_new_id("exp"), name=name, dataset_id=dataset_id, architecture=architecture, **kw)


class Model(BaseEntity):
    """A trained, checkpointed model — the deployable artifact of an Experiment."""

    experiment_id: str
    architecture: str
    checkpoint_path: Optional[str] = None
    deployment_format: Optional[str] = None  # pytorch | onnx | tensorrt
    validation_accuracy: Optional[float] = None
    known_limitations: list[str] = Field(default_factory=list)

    @classmethod
    def create(cls, name: str, experiment_id: str, architecture: str, **kw) -> "Model":
        return cls(id=_new_id("model"), name=name, experiment_id=experiment_id, architecture=architecture, **kw)


class Signal(BaseEntity):
    """A single captured/generated RF signal instance."""

    signal_type: Optional[str] = None  # e.g. bpsk, qpsk, sweep_jammer, barrage_jammer
    frequency: Optional[float] = None
    snr_db: Optional[float] = None
    duration_s: Optional[float] = None
    quality: Optional[str] = None
    capture_time: Optional[datetime] = None
    iq_ref: Optional[str] = None  # path/URI to the raw IQ data, not the data itself

    @classmethod
    def create(cls, name: str, **kw) -> "Signal":
        return cls(id=_new_id("sig"), name=name, **kw)


class Detection(BaseEntity):
    """The event of a signal being captured and submitted for classification."""

    signal_id: str
    detected_at: datetime = Field(default_factory=_now)

    @classmethod
    def create(cls, name: str, signal_id: str, **kw) -> "Detection":
        return cls(id=_new_id("det"), name=name, signal_id=signal_id, **kw)


class Prediction(BaseEntity):
    """A model's classification output for a signal, tying Signal -> Model."""

    signal_id: str
    model_id: str
    detection_id: Optional[str] = None
    label: str
    confidence: float
    probabilities: dict = Field(default_factory=dict)
    predicted_at: datetime = Field(default_factory=_now)

    @classmethod
    def create(cls, name: str, signal_id: str, model_id: str, label: str, confidence: float, **kw) -> "Prediction":
        return cls(
            id=_new_id("pred"), name=name, signal_id=signal_id, model_id=model_id,
            label=label, confidence=confidence, **kw,
        )


class Explanation(BaseEntity):
    """Evidence explaining a Prediction (e.g. Grad-CAM heatmap, OOD score)."""

    prediction_id: str
    method: str  # e.g. "gradcam", "mahalanobis_ood"
    artifact_ref: Optional[str] = None  # path to heatmap image, etc.
    summary: Optional[str] = None
    raw_scores: dict = Field(default_factory=dict)

    @classmethod
    def create(cls, name: str, prediction_id: str, method: str, **kw) -> "Explanation":
        return cls(id=_new_id("exp2"), name=name, prediction_id=prediction_id, method=method, **kw)


class Document(BaseEntity):
    """A piece of knowledge-base content (Phase 7 ingests these)."""

    topic: Optional[str] = None
    signal_type: Optional[str] = None
    authority: Optional[str] = None
    content_ref: Optional[str] = None  # path/URI, not embedded content

    @classmethod
    def create(cls, name: str, **kw) -> "Document":
        return cls(id=_new_id("doc"), name=name, **kw)


class Evaluation(BaseEntity):
    """A recorded evaluation run over a Model or, later, an Agent (Phase 11/12)."""

    target_type: str  # "model" | "agent" | "rag"
    target_id: str
    metrics: dict = Field(default_factory=dict)

    @classmethod
    def create(cls, name: str, target_type: str, target_id: str, **kw) -> "Evaluation":
        return cls(id=_new_id("eval"), name=name, target_type=target_type, target_id=target_id, **kw)


class TrustSignal(BaseEntity):
    """Standardized trust object combining confidence + OOD + data quality
    into a single decision, per Phase 6.
    """

    prediction_id: str
    confidence: float
    ood_score: Optional[float] = None
    data_quality: Optional[float] = None
    model_status: Optional[str] = None
    evidence_count: int = 0
    trust_level: TrustLevel = TrustLevel.MEDIUM

    @classmethod
    def create(cls, name: str, prediction_id: str, confidence: float, **kw) -> "TrustSignal":
        return cls(id=_new_id("trust"), name=name, prediction_id=prediction_id, confidence=confidence, **kw)
