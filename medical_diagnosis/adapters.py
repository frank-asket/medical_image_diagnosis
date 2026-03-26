"""Domain model adapters that convert raw agent outputs into diagnostic contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

ClinicalDomain = Literal["radiology", "dermatology", "ophthalmology"]


class DomainModelAdapter(Protocol):
    """Contract for plugging in validated domain models later."""

    domain: ClinicalDomain

    def infer(self, preprocessed_image: Any, llm_result: dict[str, Any]) -> dict[str, Any]:
        """
        Return a normalized diagnosis payload.
        Expected keys:
          - diagnosis_label
          - confidence
          - triage_level
          - rationale
          - differential_diagnoses
        """


@dataclass
class HeuristicAdapter:
    """
    Temporary adapter that normalizes the LLM specialist output into a stable
    diagnosis contract until calibrated domain models are integrated.
    """

    domain: ClinicalDomain

    def infer(self, preprocessed_image: Any, llm_result: dict[str, Any]) -> dict[str, Any]:
        if self.domain == "radiology":
            label = str(llm_result.get("primary_impression", "indeterminate"))
            conf = float(llm_result.get("confidence", 0.0) or 0.0)
            ddx = llm_result.get("differential_diagnoses", []) or []
        elif self.domain == "dermatology":
            label = str(llm_result.get("classification", "indeterminate"))
            conf = float(llm_result.get("confidence", 0.0) or 0.0)
            ddx = llm_result.get("differential_diagnoses", []) or []
        else:
            label = str(llm_result.get("diagnosis_impression", "indeterminate"))
            conf = float(llm_result.get("confidence", 0.0) or 0.0)
            ddx = llm_result.get("differential_diagnoses", []) or []

        if conf >= 0.85:
            triage = "routine"
        elif conf >= 0.6:
            triage = "priority"
        else:
            triage = "urgent_review"

        return {
            "source": "adapter_heuristic_from_specialist_output",
            "diagnosis_label": label,
            "confidence": round(max(0.0, min(1.0, conf)), 4),
            "triage_level": triage,
            "rationale": str(llm_result.get("findings", "")),
            "differential_diagnoses": ddx if isinstance(ddx, list) else [str(ddx)],
        }
