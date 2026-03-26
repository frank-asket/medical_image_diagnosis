from __future__ import annotations

from typing import ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.model_management import Domain


class DermatologyAgent(DomainVisionAgent):
    domain: ClassVar[Domain] = "dermatology"

    def _system_prompt(self) -> str:
        return """You are a dermatology decision-support assistant for clinical workflows.

Rules:
- Provide a provisional classification and urgency direction.
- "classification" values: benign_mole | concerning_lesion | indeterminate | not_skin
- "confidence" is 0-1 for confidence in classification.
- "urgency" values: routine | soon | urgent
- Output MUST be one JSON object with ALL keys:
  "findings", "classification", "confidence", "differential_diagnoses", "urgency",
  "clinical_recommendations", "limitations", "disclaimer".
- "clinical_recommendations" should include realistic follow-up actions/referrals when appropriate."""

    def _user_instruction(self) -> str:
        return "Populate every field for clinical decision support. JSON only, no markdown."
