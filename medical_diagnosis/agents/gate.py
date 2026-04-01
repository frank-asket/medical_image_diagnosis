from __future__ import annotations

from typing import Any, ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.guardrails import validate_gate_output
from medical_diagnosis.model_management import Domain


class MedicalImageGateAgent(DomainVisionAgent):
    """Early gate for user-selected domains: assesses whether the image is clinical/medical."""

    domain: ClassVar[Domain] = "image_gate"

    def _system_prompt(self) -> str:
        return """You gate images for a **clinical decision-support demo** (not direct patient care).

Decide whether the input looks like a **clinical medical image** appropriate for radiology, dermatology, or ophthalmology workflows (e.g. X-ray/CT/MRI-style, skin close-up, fundus photo). Reject obvious non-medical content (selfies, landscapes, screenshots, generic objects) when you are confident.

Output MUST be exactly one JSON object:
{
  "medical_image_assessment": {
    "is_clinical_medical_image": <true|false>,
    "confidence": <number 0 to 1>,
    "category_hint": "<radiology_style|dermatology_style|ophthalmology_style|non_medical|unclear>",
    "brief_reason": "<short string>"
  }
}"""

    def _user_instruction(self) -> str:
        return "Assess this image for clinical suitability. JSON only, no markdown."

    def assess(self, image) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        raw = self.run(image)
        inner = raw.get("medical_image_assessment")
        err_source = raw if isinstance(inner, dict) else {"medical_image_assessment": inner}
        errs = validate_gate_output(err_source)
        assessment = inner if isinstance(inner, dict) else {}
        return assessment, errs, raw.get("_agent_meta", {})
