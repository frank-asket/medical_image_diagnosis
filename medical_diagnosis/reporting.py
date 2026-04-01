"""GPT-4 text layer: lay interpretation, medical-style reports, provider advice, and clinical Q&A."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

from medical_diagnosis.config import OPENAI_API_KEY, OPENAI_MODEL
from medical_diagnosis.model_management import ModelRegistry


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def diagnosis_without_meta(diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in diagnosis.items() if k != "_agent_meta"}


class DiagnosticNarrativeService:
    """
    Consumes structured outputs from vision agents and produces:
    - Lay-language interpretation
    - A more formal report-style narrative
    - Contextual follow-up / referral / next-step suggestions for providers
    """

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        registry: ModelRegistry | None = None,
        model: str | None = None,
    ) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = client or OpenAI(api_key=OPENAI_API_KEY)
        self.registry = registry or ModelRegistry()
        self.model = model or OPENAI_MODEL

    def generate_narratives(
        self,
        *,
        routing: dict[str, Any],
        diagnosis: dict[str, Any],
        patient_context: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        payload = {
            "routing": routing,
            "diagnostic_agent_output": diagnosis_without_meta(diagnosis),
        }
        if patient_context:
            payload["optional_demo_patient_context"] = patient_context

        system = """You support clinical interpretation workflows. You receive JSON from prior image agents.

Your tasks in one response:
1) **layman_interpretation**: Explain the agent output in plain language for a non-expert. Short paragraphs, no jargon unless briefly defined.
2) **medical_report**: A concise professional-style summary suitable as a handoff note (not a legal medical record). Include key findings, impression, and confidence/uncertainty as stated in the source JSON.
3) **contextual_advice**: For healthcare providers, provide practical follow-up tests, referral considerations, treatment-path considerations, and next steps when uncertainty or escalation exists. Do not invent findings not supported by the JSON. Do not prescribe specific drugs or doses.

If **optional_demo_patient_context** is present, use it for context-aware recommendations.

Hard rules:
- Do not assert new diagnoses beyond the supplied JSON.
- Output ONE JSON object with keys:
  "layman_interpretation" (string),
  "medical_report" (string),
  "contextual_advice" (object with keys: "follow_up_suggestions" (array of strings), "referral_considerations" (array of strings), "next_steps_for_provider" (array of strings), "uncertainty_notes" (string)),
  "disclaimer" (string, mention that final decisions remain with licensed clinicians)."""

        msg, usage = self._chat_json(system, json.dumps(payload, ensure_ascii=False))
        latency = round((time.perf_counter() - t0) * 1000, 2)
        self.registry.record_inference("reporting", latency)
        msg["_agent_meta"] = {
            "domain": "reporting",
            "logical_model": self.registry.get_model("reporting").name,
            "openai_model": self.model,
            "latency_ms": latency,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }
        return msg

    def answer_clinical_question(
        self,
        *,
        routing: dict[str, Any],
        diagnosis: dict[str, Any],
        narratives: dict[str, Any] | None,
        question: str,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        bundle = {
            "routing": routing,
            "diagnostic_agent_output": diagnosis_without_meta(diagnosis),
            "prior_narratives": {k: v for k, v in (narratives or {}).items() if k != "_agent_meta"} or None,
            "question": question,
        }

        system = """You answer follow-up questions from healthcare professionals about a bundled diagnostic output.

Rules:
- Ground answers in the provided JSON only; if the JSON does not support an answer, say so clearly.
- Provide educational, general information and differential thinking where appropriate; do not issue patient-specific orders.
- Output ONE JSON object with keys:
  "answer" (string),
  "caveats" (string),
  "related_topics_to_review" (array of strings, optional themes for the clinician to explore)."""

        msg, usage = self._chat_json(system, json.dumps(bundle, ensure_ascii=False))
        latency = round((time.perf_counter() - t0) * 1000, 2)
        self.registry.record_inference("clinical_qa", latency)
        msg["_agent_meta"] = {
            "domain": "clinical_qa",
            "logical_model": self.registry.get_model("clinical_qa").name,
            "openai_model": self.model,
            "latency_ms": latency,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }
        return msg

    def _chat_json(self, system: str, user_content: str) -> tuple[dict[str, Any], Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=2500,
        )
        m = resp.choices[0].message
        refusal = getattr(m, "refusal", None)
        if refusal:
            raise RuntimeError(f"Model refused narrative/QA request: {refusal}")
        raw = m.content
        if not raw or not raw.strip():
            raise RuntimeError("Empty model response for narrative/QA.")
        return json.loads(_strip_json_fence(raw)), resp.usage
