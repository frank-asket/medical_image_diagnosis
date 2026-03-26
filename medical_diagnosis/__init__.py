"""Multi-agent medical image diagnosis orchestration (GPT-4 vision + preprocessing)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medical_diagnosis.orchestrator import MedicalDiagnosisOrchestrator


def __getattr__(name: str):
    if name == "MedicalDiagnosisOrchestrator":
        from medical_diagnosis.orchestrator import MedicalDiagnosisOrchestrator

        return MedicalDiagnosisOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MedicalDiagnosisOrchestrator"]
