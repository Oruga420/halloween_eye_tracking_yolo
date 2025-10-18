"""
One Euro Filter para suavizar coordenadas en interacción en vivo.
Basado en: https://gery.casiez.net/1euro/
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


def _alpha(cutoff: float, freq: float) -> float:
    """
    Calcula el coeficiente alpha para un cutoff dado y frecuencia de muestreo.
    """
    if cutoff <= 0.0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


@dataclass
class OneEuroParams:
    """
    Parámetros del filtro One Euro.
    """

    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0


class OneEuroFilter:
    """
    Filtro One Euro adaptativo ideal para coordenadas de tracking.
    """

    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        if freq <= 0:
            raise ValueError("freq debe ser mayor a cero")
        self.freq = freq
        self.params = OneEuroParams(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """
        Limpia el estado interno.
        """
        self._x_prev = None
        self._dx_prev = 0.0
        self._last_timestamp = None

    def __call__(self, value: float, timestamp: Optional[float] = None) -> float:
        """
        Filtra un nuevo valor. Si no se provee timestamp usa time.perf_counter().
        """
        if timestamp is None:
            timestamp = time.perf_counter()
        if self._last_timestamp is None:
            self._initialize(value, timestamp)
            return value

        dt = max(timestamp - self._last_timestamp, 1e-6)
        self.freq = max(1e-3, 1.0 / dt)

        # derivada filtrada
        alpha_d = _alpha(self.params.d_cutoff, self.freq)
        dx = (value - self._x_prev) * self.freq
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        # cutoff adaptado
        cutoff = self.params.min_cutoff + self.params.beta * abs(dx_hat)
        alpha = _alpha(cutoff, self.freq)
        x_hat = alpha * value + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._last_timestamp = timestamp
        return x_hat

    def update_params(self, min_cutoff: float, beta: float) -> None:
        """
        Permite ajustar parámetros en tiempo real manteniendo estado.
        """
        self.params.min_cutoff = max(0.001, float(min_cutoff))
        self.params.beta = max(0.0, float(beta))

    def _initialize(self, value: float, timestamp: float) -> None:
        self._x_prev = value
        self._dx_prev = 0.0
        self._last_timestamp = timestamp
