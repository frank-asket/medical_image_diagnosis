from __future__ import annotations

from typing import ClassVar

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.guardrails import RouterClassification, validate_router_output
from medical_diagnosis.model_management import Domain


class DomainRouterAgent(DomainVisionAgent):
    """Classifies which clinical pipeline best fits the image."""

    domain: ClassVar[Domain] = "router"

    def _system_prompt(self) -> str:
        return """You route images for a **software testing demo** (not for patient care).

Pick the best-matching pipeline label:
- radiology: grayscale anatomical imaging style (e.g. X-ray/CT/MRI-like)
- dermatology: skin close-ups, rashes, lesions
- ophthalmology: fundus / retina-style circular color photos

Also assess whether the image is appropriate **clinical medical imaging** for these pipelines. Reject obvious non-medical content (selfies, unrelated photos) when confidence is high.

Output MUST be one JSON object with ALL keys:
{
  "domain": "<radiology|dermatology|ophthalmology>",
  "reason": "<short string>",
  "medical_image_assessment": {
    "is_clinical_medical_image": <true|false>,
    "confidence": <number 0 to 1>,
    "category_hint": "<radiology_style|dermatology_style|ophthalmology_style|non_medical|unclear>",
    "brief_reason": "<short string>"
  }
}"""

    def _user_instruction(self) -> str:
        return "Classify this image for routing and medical suitability. JSON only."

    def classify(self, image) -> RouterClassification:
        result = self.run(image)
        errs = validate_router_output(result)
        dom = result.get("domain", "radiology")
        if dom not in ("radiology", "dermatology", "ophthalmology"):
            dom = "radiology"
        reason = result.get("reason", "")
        if not isinstance(reason, str):
            reason = ""
        return RouterClassification(
            domain=dom,  # type: ignore[arg-type]
            reason=reason,
            raw=result,
            validation_errors=errs,
        )
