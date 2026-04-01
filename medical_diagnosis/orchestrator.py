"""Coordinates preprocessing, model registry, routing, and domain vision agents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from medical_diagnosis.agents import (
    DermatologyAgent,
    DomainRouterAgent,
    OphthalmologyAgent,
    RadiologyAgent,
)
from medical_diagnosis.agents.gate import MedicalImageGateAgent
from medical_diagnosis.adapters import ClinicalDomain, DomainModelAdapter, HeuristicAdapter
from medical_diagnosis.config import (
    GUARDRAILS_MEDICAL_BLOCK_MIN_CONFIDENCE,
    GUARDRAILS_NARRATIVE_MIN_CONFIDENCE,
)
from medical_diagnosis.guardrails import (
    should_block_non_medical_image,
    specialist_confidence_below_narrative_threshold,
    suppressed_narrative_placeholder,
    validate_specialist_output,
)
from medical_diagnosis.model_management import ModelRegistry
from medical_diagnosis.observability.langfuse_client import (
    get_tracer,
    hash_patient_context,
    log_generation,
    safe_diagnosis_output,
    truncate_for_trace,
    vision_descriptor,
)
from medical_diagnosis.preprocessing import ImagePreprocessor, PreprocessedImage
from medical_diagnosis.reporting import DiagnosticNarrativeService
from medical_diagnosis.security import content_fingerprint, enforce_image_size, redact_for_log

logger = logging.getLogger(__name__)

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
        adapters: dict[ClinicalDomain, DomainModelAdapter] | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.apply_clahe = apply_clahe
        self._router = DomainRouterAgent(registry=self.registry)
        self._gate = MedicalImageGateAgent(registry=self.registry)
        self._radiology = RadiologyAgent(registry=self.registry)
        self._dermatology = DermatologyAgent(registry=self.registry)
        self._ophthalmology = OphthalmologyAgent(registry=self.registry)
        self._narratives = DiagnosticNarrativeService(registry=self.registry)
        self._adapters: dict[ClinicalDomain, DomainModelAdapter] = adapters or {
            "radiology": HeuristicAdapter("radiology"),
            "dermatology": HeuristicAdapter("dermatology"),
            "ophthalmology": HeuristicAdapter("ophthalmology"),
        }

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

    @staticmethod
    def _initial_guardrails_state() -> dict[str, Any]:
        return {
            "medical_image_assessment": None,
            "router_validation_errors": None,
            "gate_validation_errors": None,
            "specialist_schema_valid": None,
            "specialist_schema_errors": None,
            "low_confidence_review_required": False,
            "narratives_suppressed": False,
            "narratives_suppressed_reason": None,
            "pipeline_status": "ok",
            "blocked_reason": None,
        }

    def _publish_guardrails(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            **state,
            "thresholds": {
                "medical_block_min_confidence": GUARDRAILS_MEDICAL_BLOCK_MIN_CONFIDENCE,
                "narrative_min_confidence": GUARDRAILS_NARRATIVE_MIN_CONFIDENCE,
            },
        }

    def _blocked_diagnosis(self, *, blocked_reason: str, detail: str) -> dict[str, Any]:
        return {
            "pipeline_blocked": True,
            "blocked_reason": blocked_reason,
            "detail": detail,
            "provisional_diagnosis": {
                "source": "guardrails",
                "diagnosis_label": "not_applicable",
                "confidence": 0.0,
                "triage_level": "urgent_review",
                "rationale": detail,
                "differential_diagnoses": [],
            },
        }

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

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

        session_id = str(uuid4())
        _obs = get_tracer()
        root_in: dict[str, Any] = {
            "mode": mode,
            "with_narratives": with_narratives,
            "has_clinical_question": bool(clinical_question),
            "image": {
                "content_fingerprint": fp,
                "pixels_logged": False,
                "note": "Vision models receive the image; Langfuse stores fingerprint only.",
            },
            "patient_context_supplied": bool(patient_context and patient_context.strip()),
        }
        if clinical_question and clinical_question.strip():
            root_in["clinical_question"] = truncate_for_trace(clinical_question.strip())

        _obs.start_trace(
            "medical-diagnosis-pipeline",
            session_id=session_id,
            metadata={
                "image_fingerprint": fp,
                "mode": mode,
                "patient_context_hash": hash_patient_context(patient_context),
            },
            input=root_in,
        )

        _trace_output: dict[str, Any] | None = None
        try:
            out = self._run_pipeline(
                path,
                mode,
                fp,
                _obs,
                with_narratives=with_narratives,
                patient_context=patient_context,
                clinical_question=clinical_question,
            )
            out["_trace_id"] = _obs.trace_id
            out["_session_id"] = session_id
            _trace_output = safe_diagnosis_output(out)
            return out
        finally:
            _obs.end_trace(output=_trace_output)
            _obs.flush()

    def _run_pipeline(
        self,
        path: Path,
        mode: RunMode,
        fp: str,
        _obs: Any,
        *,
        with_narratives: bool,
        patient_context: str | None,
        clinical_question: str | None,
    ) -> dict[str, Any]:
        """Inner pipeline logic, factored out so the trace always closes."""
        gstate = self._initial_guardrails_state()
        router_pre = ImagePreprocessor(
            target_size=(224, 224), apply_clahe=self.apply_clahe
        ).process_path(path)

        models_touched: dict[str, dict[str, str]] = {}
        routing_info: dict[str, Any]
        routed_domain: ClinicalDomain
        route_reason: str

        # ── Routing / image gate ──────────────────────────────────────

        if mode == "auto":
            router_vision = vision_descriptor(
                fingerprint=fp,
                width=router_pre.width,
                height=router_pre.height,
                channels=router_pre.channels,
                stage="router_preprocess",
            )
            with _obs.span(
                "domain-routing",
                metadata={"mode": "auto"},
                input={"vision": router_vision, "task": "domain_routing_and_image_gate"},
            ) as _rs:
                rc = self._router.classify(router_pre)
                r_model = self.registry.get_model("router")
                router_out = {
                    "domain": rc.domain,
                    "has_validation_errors": bool(rc.validation_errors),
                    "model": r_model.name,
                    "reason_preview": truncate_for_trace(rc.reason, 2000),
                }
                _rs.update(output=router_out)
                log_generation(
                    _rs,
                    rc.raw.get("_agent_meta", {}),
                    input_summary={"vision": router_vision, "task": "domain_routing_and_image_gate"},
                    output_summary={
                        "domain": rc.domain,
                        "reason_preview": truncate_for_trace(rc.reason, 2000),
                    },
                )

            models_touched["router"] = {"name": r_model.name, "version": r_model.version}
            routed_domain = rc.domain
            route_reason = rc.reason

            if rc.validation_errors:
                gstate["router_validation_errors"] = rc.validation_errors
                gstate["pipeline_status"] = "blocked"
                gstate["blocked_reason"] = "router_output_schema_validation_failed"
                logger.warning("Router output failed schema validation: %s", rc.validation_errors)
                _obs.event("guardrail-block", metadata={"reason": gstate["blocked_reason"]})
                routing_info = {"mode": mode, "domain": routed_domain, "reason": route_reason}
                return self._finish_blocked_run(
                    routing_info=routing_info,
                    router_pre=router_pre,
                    guardrails_state=gstate,
                    models_touched=models_touched,
                    diagnosis=self._blocked_diagnosis(
                        blocked_reason=gstate["blocked_reason"],
                        detail="Router JSON did not match the expected contract; specialist was not run.",
                    ),
                )

            assessment = rc.raw.get("medical_image_assessment")
            if isinstance(assessment, dict):
                gstate["medical_image_assessment"] = assessment

            block, nb_reason = should_block_non_medical_image(
                assessment if isinstance(assessment, dict) else {}
            )
            if block:
                gstate["pipeline_status"] = "blocked"
                gstate["blocked_reason"] = nb_reason
                logger.info(
                    "Pipeline blocked by medical image gate fingerprint=%s reason=%s",
                    fp,
                    nb_reason,
                )
                _obs.event("guardrail-block", metadata={"reason": nb_reason})
                routing_info = {"mode": mode, "domain": routed_domain, "reason": route_reason}
                return self._finish_blocked_run(
                    routing_info=routing_info,
                    router_pre=router_pre,
                    guardrails_state=gstate,
                    models_touched=models_touched,
                    diagnosis=self._blocked_diagnosis(
                        blocked_reason=nb_reason,
                        detail="Image assessment indicated non-clinical or non-medical content.",
                    ),
                )

            logger.info(
                "Router selected domain=%s reason=%s",
                routed_domain,
                redact_for_log(route_reason),
            )
            routing_info = {"mode": mode, "domain": routed_domain, "reason": route_reason}
        else:
            routed_domain = mode
            route_reason = "user_selected"

            gate_vision = vision_descriptor(
                fingerprint=fp,
                width=router_pre.width,
                height=router_pre.height,
                channels=router_pre.channels,
                domain=routed_domain,
                stage="image_gate_preprocess",
            )
            with _obs.span(
                "image-gate",
                metadata={"domain": routed_domain},
                input={"vision": gate_vision, "task": "clinical_image_gate"},
            ) as _gs:
                g_model = self.registry.get_model("image_gate")
                assessment, gate_errs, gate_meta = self._gate.assess(router_pre)
                gate_out = {
                    "has_errors": bool(gate_errs),
                    "model": g_model.name,
                    "is_clinical": assessment.get("is_clinical_medical_image"),
                    "category_hint": assessment.get("category_hint"),
                }
                _gs.update(output=gate_out)
                log_generation(
                    _gs,
                    gate_meta,
                    input_summary={"vision": gate_vision, "task": "clinical_image_gate"},
                    output_summary=gate_out,
                )

            models_touched["image_gate"] = {"name": g_model.name, "version": g_model.version}

            if gate_errs:
                gstate["gate_validation_errors"] = gate_errs
                gstate["pipeline_status"] = "blocked"
                gstate["blocked_reason"] = "image_gate_schema_validation_failed"
                logger.warning("Image gate output failed schema validation: %s", gate_errs)
                _obs.event("guardrail-block", metadata={"reason": gstate["blocked_reason"]})
                routing_info = {"mode": mode, "domain": routed_domain, "reason": route_reason}
                return self._finish_blocked_run(
                    routing_info=routing_info,
                    router_pre=router_pre,
                    guardrails_state=gstate,
                    models_touched=models_touched,
                    diagnosis=self._blocked_diagnosis(
                        blocked_reason=gstate["blocked_reason"],
                        detail="Image gate JSON did not match the expected contract; specialist was not run.",
                    ),
                )

            gstate["medical_image_assessment"] = assessment
            block, nb_reason = should_block_non_medical_image(assessment)
            if block:
                gstate["pipeline_status"] = "blocked"
                gstate["blocked_reason"] = nb_reason
                logger.info(
                    "Pipeline blocked by image gate fingerprint=%s reason=%s",
                    fp,
                    nb_reason,
                )
                _obs.event("guardrail-block", metadata={"reason": nb_reason})
                routing_info = {"mode": mode, "domain": routed_domain, "reason": route_reason}
                return self._finish_blocked_run(
                    routing_info=routing_info,
                    router_pre=router_pre,
                    guardrails_state=gstate,
                    models_touched=models_touched,
                    diagnosis=self._blocked_diagnosis(
                        blocked_reason=nb_reason,
                        detail="Image assessment indicated non-clinical or non-medical content.",
                    ),
                )

            routing_info = {"mode": mode, "domain": routed_domain, "reason": route_reason}

        # ── Specialist vision agent ───────────────────────────────────

        processed = self.preprocess_for_domain(path, routed_domain)
        agent = self._agent_for(routed_domain)

        spec_vision = vision_descriptor(
            fingerprint=fp,
            width=processed.width,
            height=processed.height,
            channels=processed.channels,
            domain=routed_domain,
            stage="specialist_preprocess",
        )
        with _obs.span(
            "specialist-analysis",
            metadata={"domain": routed_domain},
            input={
                "vision": spec_vision,
                "task": f"{routed_domain}_vision_specialist",
            },
        ) as _ss:
            result = agent.run(processed)
            spec_model = self.registry.get_model(routed_domain)
            spec_out = {
                "confidence": result.get("confidence"),
                "model": spec_model.name,
                "findings_preview": truncate_for_trace(result.get("findings")),
                "impression_preview": truncate_for_trace(
                    result.get("primary_impression")
                    or result.get("diagnosis_impression")
                    or result.get("classification"),
                ),
            }
            _ss.update(output=spec_out)
            log_generation(
                _ss,
                result.get("_agent_meta", {}),
                input_summary={
                    "vision": spec_vision,
                    "task": f"{routed_domain}_vision_specialist",
                },
                output_summary=spec_out,
            )

        spec_errs = validate_specialist_output(routed_domain, result)
        gstate["specialist_schema_valid"] = len(spec_errs) == 0
        gstate["specialist_schema_errors"] = spec_errs or None

        if spec_errs:
            logger.warning("Specialist output failed schema validation: %s", spec_errs)
            _obs.event(
                "guardrail-specialist-validation-failed",
                metadata={"errors": spec_errs[:5]},
            )
            result["provisional_diagnosis"] = {
                "source": "guardrails_schema_validation_failed",
                "diagnosis_label": "indeterminate",
                "confidence": 0.0,
                "triage_level": "urgent_review",
                "rationale": "Specialist vision output did not match the expected JSON schema; treat as unreliable.",
                "differential_diagnoses": [],
            }
        else:
            with _obs.span("diagnosis-adapter", metadata={"domain": routed_domain}) as _ad:
                result["provisional_diagnosis"] = self._adapters[routed_domain].infer(processed, result)
                _ad.update(output={
                    "diagnosis_label": result["provisional_diagnosis"].get("diagnosis_label"),
                    "confidence": result["provisional_diagnosis"].get("confidence"),
                    "triage_level": result["provisional_diagnosis"].get("triage_level"),
                })

        if spec_errs:
            gstate["low_confidence_review_required"] = True
        elif specialist_confidence_below_narrative_threshold(result):
            gstate["low_confidence_review_required"] = True

        suppress_reason: str | None = None
        if spec_errs:
            suppress_reason = "specialist_output_failed_schema_validation"
        elif specialist_confidence_below_narrative_threshold(result):
            suppress_reason = "specialist_confidence_below_narrative_threshold"

        narrative_block: dict[str, Any] | None = None
        specialist = self.registry.get_model(routed_domain)
        models_touched["specialist"] = {"name": specialist.name, "version": specialist.version}

        out: dict[str, Any] = {
            "routing": routing_info,
            "guardrails": self._publish_guardrails(gstate),
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

        # ── Narratives ────────────────────────────────────────────────

        if with_narratives:
            if suppress_reason:
                gstate["narratives_suppressed"] = True
                gstate["narratives_suppressed_reason"] = suppress_reason
                _obs.event("narrative-suppression", metadata={"reason": suppress_reason})
                narrative_block = suppressed_narrative_placeholder(reason=suppress_reason)
                out["results_interpretation"] = {
                    "layman_interpretation": narrative_block.get("layman_interpretation", ""),
                    "disclaimer": narrative_block.get("disclaimer", ""),
                }
                out["medical_report"] = {"report_body": narrative_block.get("medical_report", "")}
                out["contextual_advice"] = narrative_block.get("contextual_advice", {})
            else:
                prov_n = result.get("provisional_diagnosis") or {}
                narrative_in = {
                    "task": "lay_report_provider_advice",
                    "routing_domain": routing_info.get("domain"),
                    "diagnosis_label": prov_n.get("diagnosis_label"),
                    "confidence": prov_n.get("confidence"),
                    "triage_level": prov_n.get("triage_level"),
                    "patient_context_supplied": bool(patient_context and patient_context.strip()),
                }
                with _obs.span("narrative-generation", input=narrative_in) as _ns:
                    narrative_block = self._narratives.generate_narratives(
                        routing=routing_info,
                        diagnosis=result,
                        patient_context=patient_context,
                    )
                    narrative_out = {
                        "layman_preview": truncate_for_trace(narrative_block.get("layman_interpretation")),
                        "medical_report_preview": truncate_for_trace(narrative_block.get("medical_report")),
                        "has_contextual_advice": bool(narrative_block.get("contextual_advice")),
                    }
                    _ns.update(output=narrative_out)
                    log_generation(
                        _ns,
                        narrative_block.get("_agent_meta", {}),
                        input_summary=narrative_in,
                        output_summary=narrative_out,
                    )
                out["results_interpretation"] = {
                    "layman_interpretation": narrative_block.get("layman_interpretation", ""),
                    "disclaimer": narrative_block.get("disclaimer", ""),
                }
                out["medical_report"] = {"report_body": narrative_block.get("medical_report", "")}
                out["contextual_advice"] = narrative_block.get("contextual_advice", {})
                rep = self.registry.get_model("reporting")
                models_touched["reporting"] = {"name": rep.name, "version": rep.version}

        # ── Clinical Q&A ──────────────────────────────────────────────

        out["guardrails"] = self._publish_guardrails(gstate)
        qa_allowed = gstate["pipeline_status"] == "ok" and gstate["specialist_schema_valid"] is True
        if clinical_question and clinical_question.strip():
            if not qa_allowed:
                logger.info("Clinical Q&A skipped: pipeline guardrails disallow follow-up for this run.")
            else:
                qa_q = clinical_question.strip()
                qa_in = {"role": "user", "content": truncate_for_trace(qa_q)}
                with _obs.span("clinical-qa", input=qa_in) as _qs:
                    qa = self._narratives.answer_clinical_question(
                        routing=routing_info,
                        diagnosis=result,
                        narratives=narrative_block,
                        question=qa_q,
                    )
                    qa_out = {
                        "role": "assistant",
                        "answer": truncate_for_trace(qa.get("answer")),
                        "caveats": truncate_for_trace(qa.get("caveats"), 2000),
                    }
                    _qs.update(output=qa_out)
                    log_generation(
                        _qs,
                        qa.get("_agent_meta", {}),
                        input_summary=qa_in,
                        output_summary=qa_out,
                    )
                out["clinical_qa"] = qa
                qa_info = self.registry.get_model("clinical_qa")
                models_touched["clinical_qa"] = {"name": qa_info.name, "version": qa_info.version}

        out["model_management"]["models_touched"] = models_touched
        return out

    def _finish_blocked_run(
        self,
        *,
        routing_info: dict[str, Any],
        router_pre: PreprocessedImage,
        guardrails_state: dict[str, Any],
        models_touched: dict[str, dict[str, str]],
        diagnosis: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "routing": routing_info,
            "guardrails": self._publish_guardrails(guardrails_state),
            "preprocessing": {
                "width": router_pre.width,
                "height": router_pre.height,
                "channels": router_pre.channels,
                "clahe": self.apply_clahe,
            },
            "model_management": {
                "models_touched": models_touched,
                "retrain_signal": self.registry.evaluate_retrain_signal(routing_info["domain"]),
                "registry_health": self.registry.health_snapshot(),
            },
            "diagnosis": diagnosis,
        }

    # ------------------------------------------------------------------
    # Follow-up Q&A
    # ------------------------------------------------------------------

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

        gr = prior_run.get("guardrails")
        if isinstance(gr, dict):
            if gr.get("pipeline_status") == "blocked":
                raise ValueError("Follow-up Q&A is disabled for blocked pipeline runs.")
            if gr.get("specialist_schema_valid") is False:
                raise ValueError("Follow-up Q&A is disabled because specialist output failed validation.")

        _obs = get_tracer()
        parent_trace_id = prior_run.get("_trace_id")
        session_id = prior_run.get("_session_id", parent_trace_id)
        q_stripped = question.strip()
        _obs.start_trace(
            "clinical-qa-followup",
            session_id=session_id,
            metadata={
                "parent_trace_id": parent_trace_id,
                "question_length": len(q_stripped),
            },
            input={
                "follow_up_question": truncate_for_trace(q_stripped),
                "parent_trace_id": parent_trace_id,
                "note": "Prior diagnosis trace holds vision + narratives; this trace is text-only Q&A.",
            },
        )

        trace_summary: dict[str, Any] | None = None
        try:
            narratives = None
            if "results_interpretation" in prior_run and "medical_report" in prior_run:
                narratives = {
                    "layman_interpretation": prior_run["results_interpretation"].get("layman_interpretation"),
                    "medical_report": prior_run["medical_report"].get("report_body"),
                    "contextual_advice": prior_run.get("contextual_advice"),
                }

            qa_in = {"role": "user", "content": truncate_for_trace(q_stripped)}
            with _obs.span("clinical-qa", input=qa_in) as _qs:
                answer = self._narratives.answer_clinical_question(
                    routing=routing,
                    diagnosis=diagnosis,
                    narratives=narratives,
                    question=q_stripped,
                )
                qa_out = {
                    "role": "assistant",
                    "answer": truncate_for_trace(answer.get("answer")),
                    "caveats": truncate_for_trace(answer.get("caveats"), 2000),
                }
                _qs.update(output=qa_out)
                log_generation(
                    _qs,
                    answer.get("_agent_meta", {}),
                    input_summary=qa_in,
                    output_summary=qa_out,
                )

            trace_summary = {
                "follow_up_question": truncate_for_trace(q_stripped),
                "assistant": truncate_for_trace(answer.get("answer")),
                "caveats": truncate_for_trace(answer.get("caveats"), 2000),
            }
            answer["_trace_id"] = _obs.trace_id
            return answer
        finally:
            _obs.end_trace(output=trace_summary or {"answered": False})
            _obs.flush()

    def _agent_for(self, domain: ClinicalDomain):
        if domain == "radiology":
            return self._radiology
        if domain == "dermatology":
            return self._dermatology
        return self._ophthalmology
