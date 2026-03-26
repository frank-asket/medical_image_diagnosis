from __future__ import annotations

from typing import ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.model_management import Domain


class OphthalmologyAgent(DomainVisionAgent):
    domain: ClassVar[Domain] = "ophthalmology"

    def _system_prompt(self) -> str:
        return """You help engineers test a multi-agent **software demo** (not clinical use).

The image may resemble a fundus photo or not.

Rules:
- Non-diagnostic visual exercise only; no patient advice.
- "severity" enum: none | mild | moderate | severe | proliferative | indeterminate | not_retinal
- "confidence" is 0-1 for your demo-style severity guess, not a clinical grade.
- Output MUST be one JSON object with ALL keys:
  "findings", "diagnosis_impression", "severity", "confidence", "differential_diagnoses",
  "clinical_recommendations", "limitations", "disclaimer"."""

    def _user_instruction(self) -> str:
        return "Populate every field. If not eye-like, use severity not_retinal. JSON only, no markdown."
