"""Coordinates preprocessing, model registry, routing, and domain vision agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from medical_diagnosis.agents import (
    DermatologyAgent,
    DomainRouterAgent,
    OphthalmologyAgent,
    RadiologyAgent,
)
from medical_diagnosis.model_management import ModelRegistry
from medical_diagnosis.preprocessing import ImagePreprocessor, PreprocessedImage
from medical_diagnosis.reporting import DiagnosticNarrativeService
from medical_diagnosis.security import content_fingerprint, enforce_image_size, redact_for_log

logger = logging.getLogger(__name__)

ClinicalDomain = Literal["radiology", "dermatology", "ophthalmology"]
RunMode = Literal["auto"] | ClinicalDomain


class MedicalDiagnosisOrchestrator:
    """
    Multi-agent pipeline:
    1. Security / policy checks on the file
    2. Image preprocessing agent (resize, normalize, optional CLAHE)
    3. Optional domain router (GPT-4 vision) when mode is ``auto``
    4. Domain specialist agent (radiology / dermatology / ophthalmology)
    5. GPT-4 text layer (optional): lay interpretation, medical report, provider contextual advice
    6. Optional clinical Q&A grounded in the same bundle
    7. Model management bookkeeping
    """

    def __init__(
        self,
        *,
        registry: ModelRegistry | None = None,
        apply_clahe: bool = False,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.apply_clahe = apply_clahe
        self._router = DomainRouterAgent(registry=self.registry)
        self._radiology = RadiologyAgent(registry=self.registry)
        self._dermatology = DermatologyAgent(registry=self.registry)
        self._ophthalmology = OphthalmologyAgent(registry=self.registry)
        self._narratives = DiagnosticNarrativeService(registry=self.registry)

    def _preprocessor_for(self, domain: ClinicalDomain) -> ImagePreprocessor:
        size = (256, 256) if domain == "ophthalmology" else (224, 224)
        return ImagePreprocessor(target_size=size, apply_clahe=self.apply_clahe)

    def preprocess_for_domain(self, image_path: str | Path, domain: ClinicalDomain) -> PreprocessedImage:
        path = Path(image_path)
        enforce_image_size(path)
        fp = content_fingerprint(path.read_bytes())
        logger.info("Preprocessing image fingerprint=%s domain=%s", fp, domain)
        pre = self._preprocessor_for(domain)
        return pre.process_path(path)

    def run(
        self,
        image_path: str | Path,
        mode: RunMode = "auto",
        *,
        with_narratives: bool = True,
        patient_context: str | None = None,
        clinical_question: str | None = None,
    ) -> dict[str, Any]:
        path = Path(image_path)
        enforce_image_size(path)
        raw_bytes = path.read_bytes()
        fp = content_fingerprint(raw_bytes)
        logger.info("Diagnosis request fingerprint=%s mode=%s", fp, mode)

        if mode == "auto":
            router_pre = ImagePreprocessor(target_size=(224, 224), apply_clahe=self.apply_clahe).process_path(path)
            routed_domain, route_reason = self._router.classify(router_pre)
            logger.info("Router selected domain=%s reason=%s", routed_domain, redact_for_log(route_reason))
        else:
            routed_domain = mode
            route_reason = "user_selected"

        processed = self.preprocess_for_domain(path, routed_domain)
        agent = self._agent_for(routed_domain)
        result = agent.run(processed)

        specialist = self.registry.get_model(routed_domain)
        models_touched = {"specialist": {"name": specialist.name, "version": specialist.version}}
        if mode == "auto":
            r = self.registry.get_model("router")
            models_touched["router"] = {"name": r.name, "version": r.version}

        routing_info = {"mode": mode, "domain": routed_domain, "reason": route_reason}
        out: dict[str, Any] = {
            "routing": routing_info,
            "preprocessing": {
                "width": processed.width,
                "height": processed.height,
                "channels": processed.channels,
                "clahe": self.apply_clahe,
            },
            "model_management": {
                "models_touched": models_touched,
                "retrain_signal": self.registry.evaluate_retrain_signal(routed_domain),
                "registry_health": self.registry.health_snapshot(),
            },
            "diagnosis": result,
        }

        narrative_block: dict[str, Any] | None = None
        if with_narratives:
            narrative_block = self._narratives.generate_narratives(
                routing=routing_info,
                diagnosis=result,
                patient_context=patient_context,
            )
            out["results_interpretation"] = {
                "layman_interpretation": narrative_block.get("layman_interpretation", ""),
                "disclaimer": narrative_block.get("disclaimer", ""),
            }
            out["medical_report"] = {"report_body": narrative_block.get("medical_report", "")}
            out["contextual_advice"] = narrative_block.get("contextual_advice", {})
            rep = self.registry.get_model("reporting")
            models_touched["reporting"] = {"name": rep.name, "version": rep.version}

        if clinical_question and clinical_question.strip():
            qa = self._narratives.answer_clinical_question(
                routing=routing_info,
                diagnosis=result,
                narratives=narrative_block,
                question=clinical_question.strip(),
            )
            out["clinical_qa"] = qa
            qa_info = self.registry.get_model("clinical_qa")
            models_touched["clinical_qa"] = {"name": qa_info.name, "version": qa_info.version}

        out["model_management"]["models_touched"] = models_touched
        return out

    def answer_question(
        self,
        prior_run: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """
        Follow-up Q&A using a JSON object previously returned by :meth:`run`
        (must include ``routing`` and ``diagnosis``).
        """
        routing = prior_run.get("routing")
        diagnosis = prior_run.get("diagnosis")
        if not isinstance(routing, dict) or not isinstance(diagnosis, dict):
            raise ValueError("prior_run must contain 'routing' and 'diagnosis' dicts")
        narratives = None
        if "results_interpretation" in prior_run and "medical_report" in prior_run:
            narratives = {
                "layman_interpretation": prior_run["results_interpretation"].get("layman_interpretation"),
                "medical_report": prior_run["medical_report"].get("report_body"),
                "contextual_advice": prior_run.get("contextual_advice"),
            }
        return self._narratives.answer_clinical_question(
            routing=routing,
            diagnosis=diagnosis,
            narratives=narratives,
            question=question.strip(),
        )

    def _agent_for(self, domain: ClinicalDomain):
        if domain == "radiology":
            return self._radiology
        if domain == "dermatology":
            return self._dermatology
        return self._ophthalmology
