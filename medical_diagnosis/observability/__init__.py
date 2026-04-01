"""Optional Langfuse observability — zero overhead when disabled."""

from medical_diagnosis.observability.langfuse_client import (
    get_tracer,
    hash_patient_context,
    log_generation,
    safe_diagnosis_output,
    submit_clinician_feedback,
)

__all__ = [
    "get_tracer",
    "hash_patient_context",
    "log_generation",
    "safe_diagnosis_output",
    "submit_clinician_feedback",
]
