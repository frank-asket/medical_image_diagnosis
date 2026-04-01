"""Shared OpenAI vision agent logic for domain specialists."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from openai import OpenAI

from medical_diagnosis.config import OPENAI_API_KEY, OPENAI_MODEL
from medical_diagnosis.model_management import Domain, ModelRegistry
from medical_diagnosis.preprocessing import PreprocessedImage


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class DomainVisionAgent(ABC):
    domain: ClassVar[Domain]

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        registry: ModelRegistry | None = None,
        model: str | None = None,
    ) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")
        self.client = client or OpenAI(api_key=OPENAI_API_KEY)
        self.registry = registry or ModelRegistry()
        self.model = model or OPENAI_MODEL

    @abstractmethod
    def _system_prompt(self) -> str:
        pass

    @abstractmethod
    def _user_instruction(self) -> str:
        pass

    def run(self, image: PreprocessedImage) -> dict[str, Any]:
        t0 = time.perf_counter()
        info = self.registry.get_model(self.domain)
        url = f"data:{image.mime_type};base64,{image.base64_data}"
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._user_instruction()},
                        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                    ],
                },
            ],
            max_tokens=1200,
        )
        msg = resp.choices[0].message
        refusal = getattr(msg, "refusal", None)
        if refusal:
            raise RuntimeError(
                "The model refused this request (often due to safety policies on medical-style images). "
                f"Refusal: {refusal}"
            )
        raw = msg.content
        if not raw or not raw.strip():
            raise RuntimeError("Empty model response; try a different image, model, or prompt framing.")
        try:
            parsed = json.loads(_strip_json_fence(raw))
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned non-JSON: {raw[:500]}") from e
        usage = resp.usage
        parsed["_agent_meta"] = {
            "domain": self.domain,
            "logical_model": info.name,
            "logical_version": info.version,
            "openai_model": self.model,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }
        self.registry.record_inference(self.domain, parsed["_agent_meta"]["latency_ms"])
        return parsed
