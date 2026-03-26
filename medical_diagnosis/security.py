"""Security helpers aligned with HIPAA/GDPR-style practices (defense in depth; not legal advice)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from medical_diagnosis.config import MAX_IMAGE_BYTES

logger = logging.getLogger(__name__)


class ImageAccessError(PermissionError):
    """Raised when an image fails policy checks."""


def enforce_image_size(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ImageAccessError(
            f"Image exceeds maximum allowed size ({MAX_IMAGE_BYTES} bytes). "
            "Configure MAX_IMAGE_BYTES if appropriate for your environment."
        )


def content_fingerprint(data: bytes) -> str:
    """Non-reversible hash for audit logs (never log raw image bytes)."""
    return hashlib.sha256(data).hexdigest()[:16]


def redact_for_log(message: str, max_len: int = 200) -> str:
    """Truncate user-facing strings before logging."""
    if len(message) <= max_len:
        return message
    return message[: max_len - 3] + "..."
