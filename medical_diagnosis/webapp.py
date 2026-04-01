"""FastAPI UI and API for image diagnosis + expert Q&A + clinician feedback."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from medical_diagnosis.observability.langfuse_client import submit_clinician_feedback
from medical_diagnosis.orchestrator import MedicalDiagnosisOrchestrator

app = FastAPI(title="Medical Image Diagnosis UI")
orchestrator = MedicalDiagnosisOrchestrator()
_bundles: dict[str, dict[str, Any]] = {}
_bundle_traces: dict[str, str | None] = {}
_bundle_sessions: dict[str, str | None] = {}
_ui_path = Path(__file__).resolve().parent / "ui" / "index.html"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_ui_path)


@app.post("/api/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    domain: str = Form("auto"),
    patient_context: str = Form(""),
) -> dict[str, Any]:
    if domain not in ("auto", "radiology", "dermatology", "ophthalmology"):
        raise HTTPException(status_code=400, detail="Invalid domain")
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = Path(tmp.name)
    try:
        result = orchestrator.run(
            tmp_path,
            mode=domain,  # type: ignore[arg-type]
            with_narratives=True,
            patient_context=patient_context.strip() or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    bundle_id = str(uuid4())
    trace_id = result.pop("_trace_id", None)
    session_id = result.pop("_session_id", None)
    _bundles[bundle_id] = result
    _bundle_traces[bundle_id] = trace_id
    _bundle_sessions[bundle_id] = session_id
    return {"bundle_id": bundle_id, "trace_id": trace_id, "result": result}


@app.post("/api/qa")
async def qa(bundle_id: str = Form(...), question: str = Form(...)) -> dict[str, Any]:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    prior = _bundles.get(bundle_id)
    if prior is None:
        raise HTTPException(status_code=404, detail="Unknown bundle_id. Upload and diagnose first.")

    parent_trace_id = _bundle_traces.get(bundle_id)
    if parent_trace_id:
        prior.setdefault("_trace_id", parent_trace_id)
    session_id = _bundle_sessions.get(bundle_id)
    if session_id:
        prior.setdefault("_session_id", session_id)

    try:
        answer = orchestrator.answer_question(prior, question.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer.pop("_trace_id", None)
    return {"bundle_id": bundle_id, "clinical_qa": answer}


@app.post("/api/voice")
async def voice(bundle_id: str = Form(...), question: str = Form(...)) -> dict[str, Any]:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")
    prior = _bundles.get(bundle_id)
    if prior is None:
        raise HTTPException(status_code=404, detail="Unknown bundle_id. Upload and diagnose first.")
    try:
        answer = orchestrator.answer_question(prior, question.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"bundle_id": bundle_id, "clinical_qa": answer}


# ── Clinician feedback endpoint ───────────────────────────────────────────


class ClinicianFeedbackRequest(BaseModel):
    """Structured clinician feedback on a diagnostic run."""

    bundle_id: str
    agreement: str = Field(
        ...,
        description="agree | partially_agree | disagree",
        pattern=r"^(agree|partially_agree|disagree)$",
    )
    corrected_diagnosis: str | None = Field(
        None, max_length=500, description="Optional corrected diagnosis label"
    )
    corrected_triage: str | None = Field(
        None, description="Optional corrected triage level"
    )
    confidence_override: float | None = Field(
        None, ge=0.0, le=1.0, description="Clinician confidence (0-1)"
    )
    comment: str | None = Field(
        None, max_length=2000, description="Free-text comment (no PHI)"
    )


@app.post("/api/feedback")
async def clinician_feedback(req: ClinicianFeedbackRequest) -> dict[str, Any]:
    """Accept structured clinician feedback and persist it via Langfuse scores.

    The feedback is linked to the original diagnosis trace. No raw images or
    PHI are stored. If Langfuse is disabled, the endpoint returns a 200 with
    ``feedback_stored: false`` so the UI can display an appropriate message.
    """
    if req.bundle_id not in _bundles:
        raise HTTPException(status_code=404, detail="Unknown bundle_id")

    trace_id = _bundle_traces.get(req.bundle_id)
    if not trace_id:
        return {
            "bundle_id": req.bundle_id,
            "feedback_stored": False,
            "detail": "No trace_id for this bundle (Langfuse may be disabled).",
        }

    stored = submit_clinician_feedback(
        trace_id,
        agreement=req.agreement,
        corrected_diagnosis=req.corrected_diagnosis,
        corrected_triage=req.corrected_triage,
        confidence_override=req.confidence_override,
        comment=req.comment,
    )

    return {
        "bundle_id": req.bundle_id,
        "feedback_stored": stored,
        "trace_id": trace_id,
    }
