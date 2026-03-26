from __future__ import annotations

from typing import ClassVar, Literal

from medical_diagnosis.agents.base import DomainVisionAgent
from medical_diagnosis.model_management import Domain


Routing = Literal["radiology", "dermatology", "ophthalmology"]


class DomainRouterAgent(DomainVisionAgent):
    """Classifies which clinical pipeline best fits the image."""

    domain: ClassVar[Domain] = "router"

    def _system_prompt(self) -> str:
        return """You route images for a **software testing demo** (not for patient care).

Pick the best-matching pipeline label:
- radiology: grayscale anatomical imaging style (e.g. X-ray/CT/MRI-like)
- dermatology: skin close-ups, rashes, lesions
- ophthalmology: fundus / retina-style circular color photos

Output MUST be one JSON object: {"domain": "<radiology|dermatology|ophthalmology>", "reason": "<short string>"}"""

    def _user_instruction(self) -> str:
        return "Classify this image for routing. JSON only."

    def classify(self, image) -> tuple[Routing, str]:
        result = self.run(image)
        dom = result.get("domain", "radiology")
        reason = result.get("reason", "")
        if dom not in ("radiology", "dermatology", "ophthalmology"):
            dom = "radiology"
        return dom, reason  # type: ignore[return-value]
