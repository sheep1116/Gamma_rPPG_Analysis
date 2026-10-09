from __future__ import annotations

import numpy as np


def make_gamma_table(gamma_value: float) -> np.ndarray:
    if gamma_value <= 0:
        raise ValueError("gamma_value must be positive")
    table = (np.arange(256, dtype=np.float32) / 255.0) ** float(gamma_value) * 255.0
    return np.clip(table, 0, 255).astype(np.uint8)


def apply_gamma_pixels(pixels: np.ndarray, gamma_table: np.ndarray) -> np.ndarray:
    """Apply a Gamma LUT elementwise to an RGB image or N×3 pixel array."""
    pixels_u8 = np.asarray(pixels, dtype=np.uint8)
    return np.asarray(gamma_table, dtype=np.uint8)[pixels_u8]


def encoded_luma(pixels: np.ndarray) -> np.ndarray:
    """Compute encoded-RGB luma Y' from RGB-ordered pixels."""
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.size == 0:
        return np.empty(0, dtype=np.float32)
    return 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
