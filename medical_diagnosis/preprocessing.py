"""Image preprocessing agent: resize, normalize, optional enhancement (OpenCV / PIL / NumPy)."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


@dataclass
class PreprocessedImage:
    """Processed image ready for vision models and optional CNN-style pipelines."""

    width: int
    height: int
    channels: int
    normalized_array: np.ndarray  # float32, shape (H, W, C), values in [0, 1]
    mime_type: str
    base64_data: str  # raw base64 without data URL prefix


def _load_raw(path: Path) -> tuple[np.ndarray, Literal["BGR", "RGB"]]:
    suffix = path.suffix.lower()
    if suffix == ".dcm":
        if pydicom is None:
            raise RuntimeError("pydicom is required for DICOM files. Install with: pip install pydicom")
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if arr.max() > 1:
            arr = arr / float(arr.max())
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] != 3:
            arr = arr[:, :, :3]
        arr_u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return arr_u8, "RGB"
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        pil = Image.open(path).convert("RGB")
        rgb = np.array(pil)
        return rgb, "RGB"
    return bgr, "BGR"


class ImagePreprocessor:
    """Prepare images for domain agents: consistent size, normalization, optional CLAHE."""

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        apply_clahe: bool = False,
    ) -> None:
        self.target_size = target_size
        self.apply_clahe = apply_clahe

    def process_path(self, image_path: str | Path) -> PreprocessedImage:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        raw, space = _load_raw(path)
        if space == "BGR":
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        else:
            rgb = raw
        return self._from_rgb(rgb, mime_type="image/jpeg")

    def process_bytes(self, data: bytes, mime_hint: str = "image/jpeg") -> PreprocessedImage:
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            rgb = np.array(pil)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._from_rgb(rgb, mime_type=mime_hint.split(";")[0].strip() or "image/jpeg")

    def _from_rgb(self, rgb: np.ndarray, mime_type: str) -> PreprocessedImage:
        h, w = self.target_size[1], self.target_size[0]
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        if self.apply_clahe and resized.ndim == 3:
            lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            resized = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2RGB)
        f32 = resized.astype(np.float32) / 255.0
        pil_img = Image.fromarray(resized)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=92)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return PreprocessedImage(
            width=w,
            height=h,
            channels=3,
            normalized_array=f32,
            mime_type="image/jpeg",
            base64_data=b64,
        )
