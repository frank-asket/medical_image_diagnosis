"""FastAPI UI and API for image diagnosis + expert Q&A."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from medical_diagnosis.orchestrator import MedicalDiagnosisOrchestrator

app = FastAPI(title="Medical Image Diagnosis UI")
orchestrator = MedicalDiagnosisOrchestrator()
_bundles: dict[str, dict[str, Any]] = {}
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
    _bundles[bundle_id] = result
    return {"bundle_id": bundle_id, "result": result}


@app.post("/api/qa")
async def qa(bundle_id: str = Form(...), question: str = Form(...)) -> dict[str, Any]:
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
