"""
Utilidades para calibración de homografía cámara → proyector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def compute_homography(src_points: Iterable[Point], dst_points: Iterable[Point]) -> np.ndarray:
    """
    Calcula una homografía 3x3 a partir de al menos 4 puntos.
    """
    src = np.array(list(src_points), dtype=np.float32)
    dst = np.array(list(dst_points), dtype=np.float32)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        raise ValueError("Se requieren al menos 4 puntos para la homografía.")
    H, status = cv2.findHomography(src, dst, cv2.RANSAC)
    if H is None:
        raise RuntimeError("No fue posible estimar la homografía.")
    return H


def save_homography(path: str | Path, H: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, H)


def load_homography(path: str | Path) -> np.ndarray:
    H = np.load(Path(path))
    if H.shape != (3, 3):
        raise ValueError("La homografía debe ser una matriz 3x3.")
    return H


def apply_homography(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """
    Aplica la homografía y retorna coordenadas proyectadas.
    """
    point = np.array([x, y, 1.0], dtype=np.float32)
    mapped = H @ point
    if mapped[2] == 0:
        return float(mapped[0]), float(mapped[1])
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])
