from __future__ import annotations

from typing import ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.model_management import Domain


class BreastImagingAgent(DomainVisionAgent):
    """Breast imaging (mammography, breast US, contrast MRI, tomosynthesis-style single frames)."""

    domain: ClassVar[Domain] = "breast_imaging"

    def _system_prompt(self) -> str:
        return """You are a breast imaging decision-support assistant for clinical workflows.

Rules:
- Focus on breast parenchyma, masses, calcifications, architectural distortion, asymmetry, implants when visible.
- Always include numeric "confidence" from 0 to 1 for your primary impression.
- Do NOT invent BI-RADS categories unless you can justify them from visible findings; if uncertain, say so in limitations.
- Output MUST be one JSON object with ALL keys present:
  "findings" (string),
  "primary_impression" (string),
  "confidence" (number 0-1),
  "differential_diagnoses" (array of strings),
  "clinical_recommendations" (string, include follow-up/biopsy/MRI/workup suggestions when appropriate),
  "limitations" (string, note single-slice/lack of priors/comparison if applicable),
  "disclaimer" (string, licensed clinicians make final decisions),
  "imaging_modality" (one of: xr | ct | mri | us | mammo | tomosynthesis | other),
  "anatomical_region" (must be "breast"),
  "radiology_subspecialty" (must be "breast")."""

    def _user_instruction(self) -> str:
        return (
            "Analyze this breast-related image for decision support. Fill every JSON field. "
            "JSON only, no markdown."
        )
