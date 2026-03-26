from __future__ import annotations

from typing import ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.model_management import Domain


class DermatologyAgent(DomainVisionAgent):
    domain: ClassVar[Domain] = "dermatology"

    def _system_prompt(self) -> str:
        return """You help engineers test a multi-agent **software demo** (not clinical use).

The image may or may not show skin.

Rules:
- Non-diagnostic: describe appearance only; do not tell someone they have or do not have cancer.
- "classification" is a coarse demo label: benign_mole | concerning_lesion | indeterminate | not_skin
- "confidence" is 0-1 for how confident you are in that demo label (not medical certainty).
- "urgency" is only a placeholder enum: routine | soon | urgent (always remind that real care needs a clinician).
- Output MUST be one JSON object with ALL keys:
  "findings", "classification", "confidence", "differential_diagnoses", "urgency",
  "clinical_recommendations", "limitations", "disclaimer"."""

    def _user_instruction(self) -> str:
        return "Populate every field for this demo pipeline. JSON only, no markdown."
