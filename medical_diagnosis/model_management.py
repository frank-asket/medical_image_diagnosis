"""Model management agent: versioning, health metadata, and simple performance bookkeeping."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

Domain = Literal[
    "radiology",
    "dermatology",
    "ophthalmology",
    "router",
    "image_gate",
    "reporting",
    "clinical_qa",
]


@dataclass
class ModelInfo:
    name: str
    version: str
    backend: str
    description: str


@dataclass
class ModelRegistry:
    """
    Tracks which logical model each agent uses. Swap versions here when you deploy
    new weights or change the OpenAI deployment name.
    """

    models: dict[Domain, ModelInfo] = field(
        default_factory=lambda: {
            "radiology": ModelInfo(
                name="radiology-vision",
                version="1.0.0",
                backend="openai-chat-vision",
                description="GPT-4 class vision for chest X-ray / CT / MRI-style screening assistance",
            ),
            "dermatology": ModelInfo(
                name="dermatology-vision",
                version="1.0.0",
                backend="openai-chat-vision",
                description="GPT-4 class vision for skin lesion triage-style assistance",
            ),
            "ophthalmology": ModelInfo(
                name="ophthalmology-vision",
                version="1.0.0",
                backend="openai-chat-vision",
                description="GPT-4 class vision for fundus / retinal image triage-style assistance",
            ),
            "router": ModelInfo(
                name="domain-router",
                version="1.1.0",
                backend="openai-chat-vision",
                description="Routes images to radiology / dermatology / ophthalmology pipeline",
            ),
            "image_gate": ModelInfo(
                name="medical-image-gate",
                version="1.0.0",
                backend="openai-chat-vision",
                description="Rejects obvious non-medical images before specialist vision when domain is user-selected",
            ),
            "reporting": ModelInfo(
                name="diagnostic-narrative",
                version="1.0.0",
                backend="openai-chat",
                description="GPT-4 text: lay summaries, reports, provider-oriented next-step suggestions from agent JSON",
            ),
            "clinical_qa": ModelInfo(
                name="clinical-follow-up-qa",
                version="1.0.0",
                backend="openai-chat",
                description="GPT-4 text: clarifications for professionals grounded in prior diagnostic bundle",
            ),
        }
    )
    inference_counts: dict[Domain, int] = field(default_factory=dict)
    last_latency_ms: dict[Domain, float] = field(default_factory=dict)

    def get_model(self, domain: Domain) -> ModelInfo:
        return self.models[domain]

    def record_inference(self, domain: Domain, latency_ms: float) -> None:
        self.inference_counts[domain] = self.inference_counts.get(domain, 0) + 1
        self.last_latency_ms[domain] = latency_ms

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "models": {k: {"name": v.name, "version": v.version, "backend": v.backend} for k, v in self.models.items()},
            "inference_counts": dict(self.inference_counts),
            "last_latency_ms": dict(self.last_latency_ms),
            "timestamp_unix": time.time(),
        }

    def evaluate_retrain_signal(self, domain: Domain, min_inferences: int = 1000) -> dict[str, Any]:
        """
        Placeholder for MLflow-style monitoring: in production, wire accuracy/precision
        from a validation set and emit alerts when below threshold.
        """
        count = self.inference_counts.get(domain, 0)
        needs_review = count >= min_inferences and count % min_inferences == 0
        return {
            "domain": domain,
            "alert": "scheduled_review" if needs_review else None,
            "message": "Consider validating against a labeled holdout set when inference volume is high.",
        }
