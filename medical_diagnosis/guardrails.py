"""Pipeline guardrails: medical-image gating, schema validation, and confidence checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from medical_diagnosis.adapters import ClinicalDomain, SpecialistDomain
from medical_diagnosis.config import (
    GUARDRAILS_MEDICAL_BLOCK_MIN_CONFIDENCE,
    GUARDRAILS_NARRATIVE_MIN_CONFIDENCE,
)

Routing = Literal["radiology", "dermatology", "ophthalmology"]


@dataclass(frozen=True)
class RouterClassification:
    domain: Routing
    reason: str
    raw: dict[str, Any]
    validation_errors: list[str]


MEDICAL_CATEGORY_HINTS = frozenset(
    {
        "radiology_style",
        "dermatology_style",
        "ophthalmology_style",
        "non_medical",
        "unclear",
    }
)

DERMATOLOGY_CLASSIFICATIONS = frozenset({"benign_mole", "concerning_lesion", "indeterminate", "not_skin"})
DERMATOLOGY_URGENCY = frozenset({"routine", "soon", "urgent"})
OPHTHALMOLOGY_SEVERITY = frozenset(
    {"none", "mild", "moderate", "severe", "proliferative", "indeterminate", "not_retinal"}
)

ROUTER_RADIOLOGY_SUBSPECIALTY = frozenset({"general", "breast", "neuro", "unclear"})

RADIOLOGY_MODALITIES = frozenset(
    {
        "xr",
        "ct",
        "mri",
        "us",
        "mammo",
        "tomosynthesis",
        "nuclear",
        "pet",
        "angiography",
        "other",
    }
)

RADIOLOGY_REGIONS = frozenset(
    {
        "breast",
        "brain",
        "spine",
        "chest",
        "abdomen_pelvis",
        "head_neck",
        "extremity",
        "other",
    }
)

RADIOLOGY_OUTPUT_SUBSPECIALTY_BY_AGENT: dict[str, str] = {
    "radiology": "general",
    "breast_imaging": "breast",
    "neuro_imaging": "neuro",
}


def _is_non_empty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _coerce_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def validate_medical_image_assessment(obj: Any) -> list[str]:
    """Validate the nested object returned by the router or image gate."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["medical_image_assessment must be an object"]
    clinical = obj.get("is_clinical_medical_image")
    if not isinstance(clinical, bool):
        errs.append("medical_image_assessment.is_clinical_medical_image must be a boolean")

    conf = _coerce_float(obj.get("confidence"))
    if conf is None:
        errs.append("medical_image_assessment.confidence must be a number")
    elif not 0.0 <= conf <= 1.0:
        errs.append("medical_image_assessment.confidence must be between 0 and 1")

    hint = obj.get("category_hint")
    if hint not in MEDICAL_CATEGORY_HINTS:
        errs.append(
            f"medical_image_assessment.category_hint must be one of {sorted(MEDICAL_CATEGORY_HINTS)}"
        )

    if not _is_non_empty_str(obj.get("brief_reason")):
        errs.append("medical_image_assessment.brief_reason must be a non-empty string")

    return errs


def validate_radiology_subspecialty_route_block(obj: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["radiology_subspecialty_route must be an object when domain is radiology"]
    sub = obj.get("subspecialty")
    if sub not in ROUTER_RADIOLOGY_SUBSPECIALTY:
        errs.append(
            f"radiology_subspecialty_route.subspecialty must be one of {sorted(ROUTER_RADIOLOGY_SUBSPECIALTY)}"
        )
    conf = _coerce_float(obj.get("confidence"))
    if conf is None or not 0.0 <= conf <= 1.0:
        errs.append("radiology_subspecialty_route.confidence must be a number from 0 to 1")
    if not _is_non_empty_str(obj.get("brief_reason")):
        errs.append("radiology_subspecialty_route.brief_reason must be a non-empty string")
    return errs


def validate_router_output(raw: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    dom = raw.get("domain")
    if dom not in ("radiology", "dermatology", "ophthalmology"):
        errs.append("router.domain must be radiology, dermatology, or ophthalmology")
    if not _is_non_empty_str(raw.get("reason")):
        errs.append("router.reason must be a non-empty string")
    assessment = raw.get("medical_image_assessment")
    errs.extend([f"router.{e}" for e in validate_medical_image_assessment(assessment)])

    rsr = raw.get("radiology_subspecialty_route")
    if dom == "radiology":
        errs.extend([f"router.{e}" for e in validate_radiology_subspecialty_route_block(rsr)])
    elif rsr is not None:
        errs.append("router.radiology_subspecialty_route must be null when domain is not radiology")
    return errs


def validate_gate_output(raw: dict[str, Any]) -> list[str]:
    assessment = raw.get("medical_image_assessment")
    if not isinstance(assessment, dict):
        return ["top-level medical_image_assessment object is required"]
    return validate_medical_image_assessment(assessment)


def _need_str_field(d: dict[str, Any], key: str, path: str, errs: list[str]) -> None:
    if not _is_non_empty_str(d.get(key)):
        errs.append(f"{path}.{key} must be a non-empty string")


def _need_number_01(d: dict[str, Any], key: str, path: str, errs: list[str]) -> None:
    v = _coerce_float(d.get(key))
    if v is None or not 0.0 <= v <= 1.0:
        errs.append(f"{path}.{key} must be a number from 0 to 1")


def _need_str_list(d: dict[str, Any], key: str, path: str, errs: list[str]) -> None:
    v = d.get(key)
    if not isinstance(v, list) or not v:
        errs.append(f"{path}.{key} must be a non-empty array")
        return
    for i, item in enumerate(v):
        if not _is_non_empty_str(item):
            errs.append(f"{path}.{key}[{i}] must be a non-empty string")
            break


def _validate_radiology_family(
    meta_skipped: dict[str, Any],
    path: str,
    errs: list[str],
    *,
    expected_subspecialty: str,
) -> None:
    _need_str_field(meta_skipped, "findings", path, errs)
    _need_str_field(meta_skipped, "primary_impression", path, errs)
    _need_number_01(meta_skipped, "confidence", path, errs)
    _need_str_list(meta_skipped, "differential_diagnoses", path, errs)
    _need_str_field(meta_skipped, "clinical_recommendations", path, errs)
    _need_str_field(meta_skipped, "limitations", path, errs)
    _need_str_field(meta_skipped, "disclaimer", path, errs)
    mod = meta_skipped.get("imaging_modality")
    if mod not in RADIOLOGY_MODALITIES:
        errs.append(
            f"{path}.imaging_modality must be one of {sorted(RADIOLOGY_MODALITIES)}"
        )
    region = meta_skipped.get("anatomical_region")
    if region not in RADIOLOGY_REGIONS:
        errs.append(f"{path}.anatomical_region must be one of {sorted(RADIOLOGY_REGIONS)}")
    sub = meta_skipped.get("radiology_subspecialty")
    if sub != expected_subspecialty:
        errs.append(
            f"{path}.radiology_subspecialty must be '{expected_subspecialty}' for this specialist"
        )


def validate_specialist_output(domain: SpecialistDomain, raw: dict[str, Any]) -> list[str]:
    """Validate specialist JSON (excluding _agent_meta) before adapters and narratives."""
    errs: list[str] = []
    meta_skipped = {k: v for k, v in raw.items() if k != "_agent_meta"}
    if domain in RADIOLOGY_OUTPUT_SUBSPECIALTY_BY_AGENT:
        exp = RADIOLOGY_OUTPUT_SUBSPECIALTY_BY_AGENT[domain]
        _validate_radiology_family(meta_skipped, domain, errs, expected_subspecialty=exp)
    elif domain == "dermatology":
        _need_str_field(meta_skipped, "findings", "dermatology", errs)
        cls = meta_skipped.get("classification")
        if cls not in DERMATOLOGY_CLASSIFICATIONS:
            errs.append(
                f"dermatology.classification must be one of {sorted(DERMATOLOGY_CLASSIFICATIONS)}"
            )
        _need_number_01(meta_skipped, "confidence", "dermatology", errs)
        _need_str_list(meta_skipped, "differential_diagnoses", "dermatology", errs)
        urg = meta_skipped.get("urgency")
        if urg not in DERMATOLOGY_URGENCY:
            errs.append(f"dermatology.urgency must be one of {sorted(DERMATOLOGY_URGENCY)}")
        _need_str_field(meta_skipped, "clinical_recommendations", "dermatology", errs)
        _need_str_field(meta_skipped, "limitations", "dermatology", errs)
        _need_str_field(meta_skipped, "disclaimer", "dermatology", errs)
    else:
        _need_str_field(meta_skipped, "findings", "ophthalmology", errs)
        _need_str_field(meta_skipped, "diagnosis_impression", "ophthalmology", errs)
        sev = meta_skipped.get("severity")
        if sev not in OPHTHALMOLOGY_SEVERITY:
            errs.append(
                f"ophthalmology.severity must be one of {sorted(OPHTHALMOLOGY_SEVERITY)}"
            )
        _need_number_01(meta_skipped, "confidence", "ophthalmology", errs)
        _need_str_list(meta_skipped, "differential_diagnoses", "ophthalmology", errs)
        _need_str_field(meta_skipped, "clinical_recommendations", "ophthalmology", errs)
        _need_str_field(meta_skipped, "limitations", "ophthalmology", errs)
        _need_str_field(meta_skipped, "disclaimer", "ophthalmology", errs)
    return errs


def resolve_radiology_subspecialty(
    *,
    routed_domain: ClinicalDomain,
    mode: str,
    user_override: str | None,
    router_raw: dict[str, Any] | None,
) -> tuple[str, str]:
    """
    Pick general | breast | neuro for radiology routing.
    Returns (resolved_subspecialty, source) where source is user_override | router | default.
    """
    if routed_domain != "radiology":
        return "general", "not_applicable"

    uo = (user_override or "").strip().lower()
    if uo in ("breast", "neuro", "general"):
        return uo, "user_override"

    if mode == "auto" and router_raw:
        block = router_raw.get("radiology_subspecialty_route")
        if isinstance(block, dict):
            sub = block.get("subspecialty")
            if sub == "breast":
                return "breast", "router"
            if sub == "neuro":
                return "neuro", "router"
            if sub == "general":
                return "general", "router"
            if sub == "unclear":
                return "general", "router_unclear_default"

    return "general", "default"


def specialist_domain_for_radiology_subspecialty(resolved: str) -> SpecialistDomain:
    if resolved == "breast":
        return "breast_imaging"
    if resolved == "neuro":
        return "neuro_imaging"
    return "radiology"


def should_block_non_medical_image(assessment: dict[str, Any]) -> tuple[bool, str]:
    """
    Returns (block, reason). Blocks when the model is sufficiently confident the
    image is not a clinical medical image.
    """
    errs = validate_medical_image_assessment(assessment)
    if errs:
        return True, "invalid_medical_image_assessment: " + "; ".join(errs)

    conf = float(assessment["confidence"])
    clinical = bool(assessment["is_clinical_medical_image"])
    hint = str(assessment["category_hint"])

    if conf >= GUARDRAILS_MEDICAL_BLOCK_MIN_CONFIDENCE:
        if not clinical:
            return True, "model_assessment_non_clinical_medical_image"
        if hint == "non_medical":
            return True, "model_assessment_category_non_medical"

    return False, ""


def specialist_confidence_below_narrative_threshold(raw: dict[str, Any]) -> bool:
    """True if the specialist-reported confidence is below the narrative layer threshold."""
    v = _coerce_float(raw.get("confidence"))
    if v is None:
        return True
    return v < GUARDRAILS_NARRATIVE_MIN_CONFIDENCE


def suppressed_narrative_placeholder(*, reason: str) -> dict[str, Any]:
    """Safe placeholder content when the narrative layer is intentionally skipped."""
    text = (
        "Automated lay summary and extended reporting were not generated. "
        f"Reason: {reason}. "
        "Have a qualified clinician review the structured specialist output (if any) before using it."
    )
    return {
        "layman_interpretation": text,
        "medical_report": text,
        "contextual_advice": {
            "follow_up_suggestions": [],
            "referral_considerations": [],
            "next_steps_for_provider": [],
            "uncertainty_notes": reason,
        },
        "disclaimer": "Final decisions remain with licensed clinicians. This output was withheld pending review.",
    }
