"""
Wrapper para Ultralytics YOLO con tracking persistente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - handled en runtime
    raise ImportError(
        "Ultralytics no está instalado. Ejecuta 'pip install -r requirements.txt'."
    ) from exc


@dataclass
class PersonDetection:
    """
    Información consolidada de una detección de persona.
    """

    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    area: float
    track_id: Optional[int] = None

    def normalized(self, frame_shape: Tuple[int, int, int]) -> "PersonDetection":
        """
        Retorna una nueva detección con centroid y bbox normalizados 0-1.
        """
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = self.bbox
        nx1, ny1 = x1 / width, y1 / height
        nx2, ny2 = x2 / width, y2 / height
        cx, cy = self.centroid
        return PersonDetection(
            centroid=(cx / width, cy / height),
            bbox=(nx1, ny1, nx2, ny2),
            confidence=self.confidence,
            area=self.area / (width * height),
            track_id=self.track_id,
        )


class PersonTracker:
    """
    Gestiona detección y tracking de personas usando YOLO11 + ByteTrack.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        device: Optional[str] = None,
        conf: float = 0.3,
        iou: float = 0.5,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou

    def process_frame(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Ejecuta detección+tracking y retorna las detecciones ordenadas por área.
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            source=frame,
            stream=False,
            persist=True,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            classes=[0],  # solo personas
            device=self.device,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

        detections: List[PersonDetection] = []
        for bbox, confidence, track_id in zip(xyxy, confs, ids):
            x1, y1, x2, y2 = bbox.tolist()
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            cx = x1 + width * 0.5
            cy = y1 + height * 0.5
            detections.append(
                PersonDetection(
                    centroid=(cx, cy),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    area=area,
                    track_id=int(track_id) if track_id is not None else None,
                )
            )

        detections.sort(key=lambda det: det.area, reverse=True)
        return detections

    def best_detection(self, frame: np.ndarray) -> Optional[PersonDetection]:
        """
        Retorna la detección con mayor área (proxy de cercanía).
        """
        detections = self.process_frame(frame)
        return detections[0] if detections else None
