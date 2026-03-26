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
