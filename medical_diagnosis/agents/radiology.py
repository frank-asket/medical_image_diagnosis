from __future__ import annotations

from typing import ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.model_management import Domain


class RadiologyAgent(DomainVisionAgent):
    domain: ClassVar[Domain] = "radiology"

    def _system_prompt(self) -> str:
        return """You help engineers test a multi-agent **software demo** (not clinical use).

The user uploads a picture that may *look like* an X-ray, CT, or MRI, or may be unrelated.

Rules:
- Treat this as a **UI/schema exercise**: describe visible patterns in neutral, non-diagnostic language.
- Do NOT provide medical advice, diagnosis, or triage for real patients.
- Always include a numeric "confidence" from 0 to 1 reflecting how well the image matches a radiology-style presentation (not disease certainty).
- Output MUST be one JSON object with ALL keys present:
  "findings" (string),
  "primary_impression" (string, tentative visual summary only),
  "confidence" (number 0-1),
  "differential_diagnoses" (array of strings; may be empty or generic educational examples),
  "clinical_recommendations" (string; must state that a qualified clinician must evaluate real studies),
  "limitations" (string),
  "disclaimer" (string; demo/educational only)."""

    def _user_instruction(self) -> str:
        return (
            "Fill every JSON field for this demo. If the image is not radiology-like, explain in limitations. "
            "Respond with JSON only, no markdown."
        )
