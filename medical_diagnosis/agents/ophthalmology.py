from __future__ import annotations

from typing import ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.model_management import Domain


class OphthalmologyAgent(DomainVisionAgent):
    domain: ClassVar[Domain] = "ophthalmology"

    def _system_prompt(self) -> str:
        return """You are an ophthalmology/retina decision-support assistant for clinical workflows.

Rules:
- Provide a provisional diagnosis impression and severity/triage orientation.
- "severity" enum: none | mild | moderate | severe | proliferative | indeterminate | not_retinal
- "confidence" is 0-1 for your primary impression.
- Output MUST be one JSON object with ALL keys:
  "findings", "diagnosis_impression", "severity", "confidence", "differential_diagnoses",
  "clinical_recommendations", "limitations", "disclaimer".
- "clinical_recommendations" should include follow-up testing/referral suggestions when warranted."""

    def _user_instruction(self) -> str:
        return "Populate every field. If not eye-like, use severity not_retinal. JSON only, no markdown."
