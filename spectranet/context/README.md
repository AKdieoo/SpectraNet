# context/

**Purpose:** the domain/context model for the AI-native platform layer.

Defines the entities that give ML outputs meaning: `Signal`, `Detection`,
`Prediction`, `Model`, `Dataset`, `Experiment`, `Explanation`, `Document`,
`Evaluation`, `TrustSignal`, and the relationships between them.

This layer does **not** call any ML code directly. It is a pure data/schema
layer that the metadata store (Phase 3), lineage (Phase 5), trust (Phase 6),
and eventually the MCP tools (Phase 9) build on top of.

Status: Phase 2 (domain model) implemented. See `models.py`.
