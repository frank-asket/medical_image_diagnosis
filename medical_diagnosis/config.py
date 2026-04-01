import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Vision-capable GPT-4 family; override with OPENAI_MODEL if needed.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


# --- Guardrails (see README “Guardrails” for tuning) ---
#
# Router / image gate return medical_image_assessment.confidence in [0, 1]. When that confidence
# is >= this value AND the assessment says non-clinical or category non_medical, the pipeline stops
# before the specialist.
#   • Too many legitimate images blocked → raise this (e.g. 0.65–0.75).
#   • Too much obvious non-medical reaching the specialist → lower this (e.g. 0.45–0.55).
GUARDRAILS_MEDICAL_BLOCK_MIN_CONFIDENCE = _clamp_0_1(
    float(os.getenv("GUARDRAILS_MEDICAL_BLOCK_MIN_CONFIDENCE", "0.6"))
)

# Specialist "confidence" compared after a successful vision call. Below this, the narrative layer
# is skipped (placeholder text only); structured output still appears if schema-valid.
#   • Narratives rarely appear / too cautious → lower (e.g. 0.4–0.45).
#   • Weak specialist confidence still getting rich narratives → raise (e.g. 0.55–0.65).
GUARDRAILS_NARRATIVE_MIN_CONFIDENCE = _clamp_0_1(
    float(os.getenv("GUARDRAILS_NARRATIVE_MIN_CONFIDENCE", "0.55"))
)

# --- Langfuse observability (optional — see LANGFUSE_INTEGRATION.md) ---
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() in ("true", "1", "yes")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
