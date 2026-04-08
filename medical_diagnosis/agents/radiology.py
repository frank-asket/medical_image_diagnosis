from __future__ import annotations

from typing import ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.model_management import Domain


class RadiologyAgent(DomainVisionAgent):
    domain: ClassVar[Domain] = "radiology"

    def _system_prompt(self) -> str:
        return """You are a radiology decision-support assistant for real clinical workflows.

Rules:
- Provide a provisional diagnosis/impression and triage orientation based on visible findings.
- Always include a numeric "confidence" from 0 to 1 for your primary impression.
- Output MUST be one JSON object with ALL keys present:
  "findings" (string),
  "primary_impression" (string),
  "confidence" (number 0-1),
  "differential_diagnoses" (array of strings),
  "clinical_recommendations" (string, include concrete next-step suggestions and urgency),
  "limitations" (string),
  "disclaimer" (string, mention that final decisions remain with licensed clinicians),
  "imaging_modality" (one of: xr | ct | mri | us | mammo | tomosynthesis | nuclear | pet | angiography | other),
  "anatomical_region" (one of: chest | abdomen_pelvis | extremity | head_neck | brain | spine | breast | other),
  "radiology_subspecialty" (must be "general" for this general radiology agent)."""

    def _user_instruction(self) -> str:
        return (
            "Fill every JSON field. If the image is not radiology-like, explain in limitations. "
            "Respond with JSON only, no markdown."
        )
