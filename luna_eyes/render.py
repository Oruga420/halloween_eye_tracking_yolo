"""
Renderiza dos ojos animados controlados por coordenadas normalizadas.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class EyeRenderConfig:
    width: int = 1280
    height: int = 720
    eye_radius: int = 150
    eye_separation: float = 0.35  # proporción del ancho
    pupil_radius: int = 45
    pupil_max_offset: float = 0.35
    easing: float = 0.15
    blink_duration: float = 0.18
    blink_interval_range: Tuple[float, float] = (3.0, 7.0)
    background_color: Tuple[int, int, int] = (0, 0, 0)
    sclera_color: Tuple[int, int, int] = (240, 240, 240)
    iris_color: Tuple[int, int, int] = (60, 140, 255)
    pupil_color: Tuple[int, int, int] = (20, 20, 20)
    highlight_color: Tuple[int, int, int] = (255, 255, 255)


class EyeRenderer:
    """
    Genera un frame listos para proyectar con ojos que siguen un objetivo.
    """

    def __init__(self, config: EyeRenderConfig) -> None:
        self.config = config
        self._current_dir = np.array([0.0, 0.0], dtype=np.float32)
        self._target_dir = np.array([0.0, 0.0], dtype=np.float32)
        self._blink_until = 0.0
        self._blink_start = 0.0
        self._next_blink = time.perf_counter() + self._random_blink_interval()
        self._blink_active = False

    def update_target(self, nx: float, ny: float) -> None:
        """
        Define el objetivo normalizado (-1, 1) a seguir.
        """
        nx = float(np.clip(nx, -1.0, 1.0))
        ny = float(np.clip(ny, -1.0, 1.0))
        self._target_dir = np.array([nx, ny], dtype=np.float32)

    def force_blink(self) -> None:
        """
        Inicia un parpadeo inmediato.
        """
        now = time.perf_counter()
        self._start_blink(now)

    def render(self) -> np.ndarray:
        """
        Retorna un frame BGR listo para proyectar.
        """
        now = time.perf_counter()
        self._update_direction(now)
        blink_ratio = self._update_blink(now)

        cfg = self.config
        frame = np.full((cfg.height, cfg.width, 3), cfg.background_color, dtype=np.uint8)

        center_y = cfg.height // 2
        half_sep = int(cfg.width * cfg.eye_separation * 0.5)
        left_center = (cfg.width // 2 - half_sep, center_y)
        right_center = (cfg.width // 2 + half_sep, center_y)

        for direction, center in ((-1, left_center), (1, right_center)):
            # Ojos miran en la misma dirección; direction se usa para simetría si se desea.
            self._draw_eye(
                frame,
                center=center,
                gaze=self._current_dir,
                blink_ratio=blink_ratio,
            )
        return frame

    def _update_direction(self, now: float) -> None:
        ease = np.clip(self.config.easing, 0.0, 1.0)
        self._current_dir = (1.0 - ease) * self._current_dir + ease * self._target_dir

    def _update_blink(self, now: float) -> float:
        if not self._blink_active and now >= self._next_blink:
            self._start_blink(now)

        if not self._blink_active:
            return 1.0

        elapsed = now - self._blink_start
        duration = self.config.blink_duration
        if duration <= 0:
            self._blink_active = False
            self._next_blink = now + self._random_blink_interval()
            return 1.0

        if elapsed >= duration:
            self._blink_active = False
            self._next_blink = now + self._random_blink_interval()
            return 1.0

        half = duration * 0.5
        if elapsed <= half:
            open_ratio = max(0.0, 1.0 - (elapsed / half))
        else:
            open_ratio = max(0.0, (elapsed - half) / half)
        return open_ratio

    def _start_blink(self, now: float) -> None:
        self._blink_active = True
        self._blink_start = now

    def _random_blink_interval(self) -> float:
        low, high = self.config.blink_interval_range
        return random.uniform(low, high)

    def _draw_eye(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        gaze: np.ndarray,
        blink_ratio: float,
    ) -> None:
        cfg = self.config
        blink_ratio = float(np.clip(blink_ratio, 0.0, 1.0))
        eye_radius = cfg.eye_radius
        gaze = np.clip(gaze, -1.0, 1.0)
        max_offset = cfg.pupil_max_offset * eye_radius
        offset = (gaze * max_offset).astype(np.int32)
        pupil_center = (int(center[0] + offset[0]), int(center[1] + offset[1]))

        sclera_axes = (eye_radius, max(1, int(eye_radius * blink_ratio)))
        cv2.ellipse(frame, center, sclera_axes, 0, 0, 360, cfg.sclera_color, -1, lineType=cv2.LINE_AA)

        iris_radius = int(eye_radius * 0.55)
        iris_axes = (iris_radius, max(1, int(iris_radius * blink_ratio)))
        cv2.ellipse(frame, pupil_center, iris_axes, 0, 0, 360, cfg.iris_color, -1, lineType=cv2.LINE_AA)

        pupil_radius = max(1, int(cfg.pupil_radius * blink_ratio))
        cv2.circle(frame, pupil_center, pupil_radius, cfg.pupil_color, -1, lineType=cv2.LINE_AA)

        highlight_pos = (
            int(pupil_center[0] - pupil_radius * 0.5),
            int(pupil_center[1] - pupil_radius * 0.5),
        )
        cv2.circle(frame, highlight_pos, max(1, pupil_radius // 3), cfg.highlight_color, -1, lineType=cv2.LINE_AA)
