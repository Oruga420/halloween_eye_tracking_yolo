"""
Módulos principales para la instalación Luna Eyes.
"""

from .tracking import PersonTracker, PersonDetection
from .filtering import OneEuroFilter
from .render import EyeRenderer, EyeRenderConfig
from .calibration import compute_homography, load_homography, save_homography

__all__ = [
    "PersonTracker",
    "PersonDetection",
    "OneEuroFilter",
    "EyeRenderer",
    "EyeRenderConfig",
    "compute_homography",
    "load_homography",
    "save_homography",
]
