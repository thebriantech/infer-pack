"""Image operation helpers shared across preprocessing / crop stages.

Centralises normalisation constants so all stages treating the same
model use identical math.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image


# ── Normalisation ────────────────────────────────────────────────────────

# fmt: off
NORM_PARAMS: dict[str, Tuple[float, float]] = {
    "scrfd":   (127.5, 128.0),   # (pixel - mean) / std
    "arcface": (127.5, 127.5),
    "0to1":    (0.0,   255.0),   # pixel / 255
}
# fmt: on


def normalize_array(
    arr: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Apply pixel normalisation in-place-safe manner.

    Parameters
    ----------
    arr : ndarray, float32, HWC
        Raw pixel values (0–255).
    mode : str
        One of ``"scrfd"``, ``"arcface"``, ``"0to1"``.

    Returns
    -------
    ndarray, float32
        Normalised array with the same shape.
    """
    mean, std = NORM_PARAMS.get(mode, (0.0, 255.0))
    return (arr - mean) / std


def rgb_to_bgr(arr: np.ndarray) -> np.ndarray:
    """Convert an HWC RGB array to BGR (reverse channel axis)."""
    return arr[:, :, ::-1]


# ── Letterbox resize ─────────────────────────────────────────────────────

def letterbox_resize(
    img: Image.Image,
    target_w: int,
    target_h: int,
) -> tuple[Image.Image, float]:
    """Resize *img* to fit inside (target_w, target_h) with black padding.

    Returns the padded PIL image and the scale factor used.
    """
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(img_resized, (0, 0))
    return canvas, scale


# ── HWC → NCHW ──────────────────────────────────────────────────────────

def hwc_to_nchw(arr: np.ndarray) -> np.ndarray:
    """Convert a HWC float32 array to NCHW (batch dim = 1)."""
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0).astype(np.float32)
