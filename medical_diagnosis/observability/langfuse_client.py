"""Langfuse observability: pipeline tracing, PHI-safe logging, clinician feedback.

When ``LANGFUSE_ENABLED`` is falsy **or** the ``langfuse`` package is not
installed, every public function in this module is a silent no-op and the
medical pipeline runs with zero overhead from observability.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from typing import Any, Generator

from medical_diagnosis.config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_ENABLED,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional SDK import – langfuse is an *optional* dependency.
# ---------------------------------------------------------------------------
_SDK_AVAILABLE = False
_get_langfuse_client = None  # type: ignore[assignment]
_propagate_attrs = None

if LANGFUSE_ENABLED:
    try:
        from langfuse import get_client as _glc  # noqa: N812
        from langfuse import propagate_attributes as _pa

        _get_langfuse_client = _glc
        _propagate_attrs = _pa
        _SDK_AVAILABLE = True
    except ImportError:
        logger.info(
            "LANGFUSE_ENABLED is set but the langfuse package is not installed. "
            "Install with: pip install langfuse"
        )

# ---------------------------------------------------------------------------
# PHI-safety utilities
# ---------------------------------------------------------------------------
_PHI_KEYS = frozenset(
    {
        "patient_name",
        "patient_id",
        "mrn",
        "dob",
        "date_of_birth",
        "ssn",
        "social_security",
        "address",
        "phone",
        "email",
        "insurance_id",
        "medical_record_number",
    }
)


def hash_patient_context(ctx: str | None) -> str | None:
    """One-way SHA-256 prefix for correlation without re-identification."""
    if not ctx:
        return None
    return hashlib.sha256(ctx.encode("utf-8")).hexdigest()[:12]


def safe_diagnosis_output(result: dict[str, Any]) -> dict[str, Any]:
    """Extract only non-PHI structured fields from a pipeline result."""
    prov = result.get("diagnosis", {}).get("provisional_diagnosis", {})
    gr = result.get("guardrails", {})
    routing = result.get("routing", {})
    return {
        "pipeline_status": gr.get("pipeline_status"),
        "blocked_reason": gr.get("blocked_reason"),
        "specialist_schema_valid": gr.get("specialist_schema_valid"),
        "narratives_suppressed": gr.get("narratives_suppressed"),
        "narratives_suppressed_reason": gr.get("narratives_suppressed_reason"),
        "low_confidence_review_required": gr.get("low_confidence_review_required"),
        "diagnosis_label": prov.get("diagnosis_label"),
        "confidence": prov.get("confidence"),
        "triage_level": prov.get("triage_level"),
        "domain": routing.get("domain"),
        "mode": routing.get("mode"),
    }


def truncate_for_trace(text: str | None, max_chars: int = 4000) -> str | None:
    """Trim long strings for Langfuse input/output (no raw image bytes)."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 16] + " … [truncated]"


def vision_descriptor(
    *,
    fingerprint: str,
    width: int,
    height: int,
    channels: int,
    domain: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Non-PHI description of an image sent to vision models (pixels never logged)."""
    d: dict[str, Any] = {
        "type": "clinical_image",
        "content_fingerprint": fingerprint,
        "dimensions_px": f"{width}x{height}",
        "channels": channels,
        "pixels_logged": False,
    }
    if domain:
        d["clinical_domain"] = domain
    if stage:
        d["pipeline_stage"] = stage
    return d


# ---------------------------------------------------------------------------
# Noop implementations (used when Langfuse is off or unavailable)
# ---------------------------------------------------------------------------


class _NoopSpan:
    """Placeholder span that silently discards all calls."""

    trace_id: str | None = None

    def update(self, **_: Any) -> "_NoopSpan":
        return self

    def end(self, **_: Any) -> None:
        pass

    def score(self, **_: Any) -> None:
        pass

    def score_trace(self, **_: Any) -> None:
        pass

    def start_observation(self, **_: Any) -> "_NoopSpan":
        return self


_NOOP = _NoopSpan()


class NoopTracer:
    """Zero-cost tracer returned when Langfuse is disabled or unavailable."""

    trace_id: str | None = None

    def start_trace(self, _name: str, **_kw: Any) -> None:
        pass

    def end_trace(self, **_kw: Any) -> None:
        pass

    @contextmanager
    def span(self, _name: str, **_kw: Any) -> Generator[_NoopSpan, None, None]:
        yield _NOOP

    def event(self, _name: str, **_kw: Any) -> None:
        pass

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Langfuse-backed tracer
# ---------------------------------------------------------------------------


class LangfuseTracer:
    """Thin, failure-safe wrapper over the Langfuse Python SDK v3+.

    Uses **explicit parent-child** via ``root.start_observation()`` rather than
    OTel context propagation, which avoids the context-token issue that causes
    child spans to become orphaned root traces when the parent is opened with
    manual ``__enter__``/``__exit__``.
    """

    def __init__(self) -> None:
        self._client = _get_langfuse_client()
        self._root_span: Any = None
        self._propagation_ctx: Any = None
        self.trace_id: str | None = None

    # -- trace (manual start / end) -----------------------------------------

    def start_trace(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        input: Any = None,  # noqa: A002
        tags: list[str] | None = None,
    ) -> None:
        """Open a root trace for one pipeline run."""
        try:
            if session_id and _propagate_attrs:
                self._propagation_ctx = _propagate_attrs(session_id=session_id)
                self._propagation_ctx.__enter__()

            self._root_span = self._client.start_observation(
                name=name, as_type="span",
            )
            update_kw: dict[str, Any] = {}
            if metadata:
                update_kw["metadata"] = metadata
            if input is not None:
                update_kw["input"] = input
            if tags:
                update_kw["tags"] = tags
            if update_kw:
                self._root_span.update(**update_kw)
            self.trace_id = _extract_trace_id(self._root_span)
        except Exception:
            logger.debug("Langfuse trace start failed for %s", name, exc_info=True)
            self._root_span = None

    def end_trace(self, *, output: Any = None) -> None:
        """Close the root trace opened by :meth:`start_trace`."""
        if self._root_span is None:
            return
        try:
            if output is not None:
                self._root_span.update(output=output)
            self._root_span.end()
        except Exception:
            logger.debug("Langfuse trace end failed", exc_info=True)
        finally:
            self._root_span = None
            if self._propagation_ctx is not None:
                try:
                    self._propagation_ctx.__exit__(None, None, None)
                except Exception:
                    pass
                self._propagation_ctx = None

    # -- child spans (explicit parent → proper nesting) ---------------------

    @contextmanager
    def span(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        input: Any = None,  # noqa: A002
        **kwargs: Any,
    ) -> Generator[Any, None, None]:
        if self._root_span is None:
            yield _NOOP
            return

        child: Any = None
        try:
            child = self._root_span.start_observation(
                name=name, as_type="span",
            )
            update_kw: dict[str, Any] = {}
            if metadata:
                update_kw["metadata"] = metadata
            if input is not None:
                update_kw["input"] = input
            if update_kw:
                child.update(**update_kw)
        except Exception:
            logger.debug("Langfuse span start failed for %s", name, exc_info=True)

        if child is None:
            yield _NOOP
            return

        try:
            yield child
        finally:
            try:
                child.end()
            except Exception:
                pass

    # -- discrete event (child of root) -------------------------------------

    def event(self, name: str, *, metadata: dict[str, Any] | None = None) -> None:
        if self._root_span is None:
            return
        try:
            evt = self._root_span.start_observation(name=name, as_type="event")
            if metadata:
                evt.update(metadata=metadata)
            evt.end()
        except Exception:
            logger.debug("Langfuse event %s failed", name, exc_info=True)

    # -- flush --------------------------------------------------------------

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.debug("Langfuse flush failed", exc_info=True)


# ---------------------------------------------------------------------------
# Trace-ID extraction (resilient to SDK version differences)
# ---------------------------------------------------------------------------


def _extract_trace_id(span: Any) -> str | None:
    for attr in ("trace_id", "_trace_id", "id"):
        val = getattr(span, attr, None)
        if val and isinstance(val, str):
            return val
    try:
        otel_ctx = getattr(span, "get_span_context", None)
        if otel_ctx:
            ctx = otel_ctx()
            if ctx and ctx.trace_id:
                return format(ctx.trace_id, "032x")
        inner = getattr(span, "_span", None) or getattr(span, "span", None)
        if inner and hasattr(inner, "get_span_context"):
            ctx = inner.get_span_context()
            if ctx and ctx.trace_id:
                return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_tracer() -> NoopTracer | LangfuseTracer:
    """Return the appropriate tracer based on config and SDK availability."""
    if LANGFUSE_ENABLED and _SDK_AVAILABLE:
        try:
            return LangfuseTracer()
        except Exception:
            logger.warning(
                "Langfuse tracer init failed; falling back to noop.",
                exc_info=True,
            )
    return NoopTracer()


# ---------------------------------------------------------------------------
# Generation logging (token/cost tracking)
# ---------------------------------------------------------------------------


def log_generation(
    parent_span: Any,
    agent_meta: dict[str, Any],
    *,
    input_summary: Any = None,
    output_summary: Any = None,
) -> None:
    """Create a ``generation`` observation under *parent_span* with model and
    token-usage data from ``_agent_meta``.  Optional *input_summary* /
    *output_summary* populate Langfuse's prompt/completion fields so traces read
    like a conversation (still PHI-minimized — no image bytes).
    """
    if isinstance(parent_span, _NoopSpan):
        return
    if not agent_meta and input_summary is None and output_summary is None:
        return
    try:
        domain = (agent_meta or {}).get("domain", "llm")
        gen = parent_span.start_observation(
            name=f"{domain}-llm-call", as_type="generation",
        )
        update_kw: dict[str, Any] = {}
        model = (agent_meta or {}).get("openai_model")
        if model:
            update_kw["model"] = model
        prompt_tokens = (agent_meta or {}).get("prompt_tokens")
        completion_tokens = (agent_meta or {}).get("completion_tokens")
        if prompt_tokens is not None or completion_tokens is not None:
            update_kw["usage_details"] = {
                "input": prompt_tokens or 0,
                "output": completion_tokens or 0,
            }
        if input_summary is not None:
            update_kw["input"] = input_summary
        if output_summary is not None:
            update_kw["output"] = output_summary
        if update_kw:
            gen.update(**update_kw)
        gen.end()
    except Exception:
        logger.debug("log_generation failed for %s", (agent_meta or {}).get("domain"), exc_info=True)


# ---------------------------------------------------------------------------
# Clinician feedback  (stored as Langfuse scores)
# ---------------------------------------------------------------------------

_AGREEMENT_SCORES: dict[str, float] = {
    "agree": 1.0,
    "partially_agree": 0.5,
    "disagree": 0.0,
}


def submit_clinician_feedback(
    trace_id: str,
    *,
    agreement: str,
    corrected_diagnosis: str | None = None,
    corrected_triage: str | None = None,
    confidence_override: float | None = None,
    comment: str | None = None,
) -> bool:
    """Persist structured clinician feedback as Langfuse scores on the trace.

    Returns ``True`` when feedback was stored, ``False`` when Langfuse is
    unavailable.  Never raises — errors are logged and swallowed so the
    clinical workflow is never interrupted.
    """
    if not (LANGFUSE_ENABLED and _SDK_AVAILABLE):
        return False

    try:
        client = _get_langfuse_client()

        score_val = _AGREEMENT_SCORES.get(agreement)
        if score_val is not None:
            client.create_score(
                trace_id=trace_id,
                name="clinician_agreement",
                value=score_val,
                data_type="NUMERIC",
                comment=f"agreement={agreement}",
            )

        client.create_score(
            trace_id=trace_id,
            name="clinician_agreement_category",
            value=agreement,
            data_type="CATEGORICAL",
        )

        if corrected_diagnosis:
            client.create_score(
                trace_id=trace_id,
                name="clinician_corrected_diagnosis",
                value=corrected_diagnosis[:500],
                data_type="CATEGORICAL",
                comment="Clinician-provided diagnosis correction",
            )

        if corrected_triage:
            client.create_score(
                trace_id=trace_id,
                name="clinician_corrected_triage",
                value=corrected_triage,
                data_type="CATEGORICAL",
            )

        if confidence_override is not None:
            clamped = max(0.0, min(1.0, float(confidence_override)))
            client.create_score(
                trace_id=trace_id,
                name="clinician_confidence_override",
                value=clamped,
                data_type="NUMERIC",
            )

        if comment:
            client.create_score(
                trace_id=trace_id,
                name="clinician_comment",
                value="provided",
                data_type="CATEGORICAL",
                comment=comment[:2000],
            )

        client.flush()
        return True
    except Exception:
        logger.warning("Failed to submit clinician feedback to Langfuse", exc_info=True)
        return False
