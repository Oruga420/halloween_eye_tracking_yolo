"""
Loop principal de la instalación: captura, tracking y render de ojos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luna_eyes.calibration import apply_homography, load_homography
from luna_eyes.filtering import OneEuroFilter
from luna_eyes.render import EyeRenderConfig, EyeRenderer
from luna_eyes.tracking import PersonDetection, PersonTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proyecta ojos que siguen a la audiencia.")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default: 0)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Modelo YOLO a usar")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo (cpu, cuda:0, mps, ...)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confianza mínima YOLO")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU mínimo YOLO")
    parser.add_argument("--homography", type=str, default=None, help="Ruta a matriz homográfica .npy")
    parser.add_argument("--output-width", type=int, default=1280, help="Ancho del lienzo proyectado")
    parser.add_argument("--output-height", type=int, default=720, help="Alto del lienzo proyectado")
    parser.add_argument("--fullscreen", action="store_true", help="Muestra la ventana de ojos en fullscreen")
    parser.add_argument("--show-camera", action="store_true", help="Muestra ventana de debug con la cámara")
    parser.add_argument("--controls", action="store_true", help="Activa sliders para ajustar filtros en vivo")
    parser.add_argument("--initial-freq", type=float, default=60.0, help="Frecuencia inicial para One Euro")
    parser.add_argument("--min-cutoff", type=float, default=1.2, help="Valor inicial min_cutoff del filtro")
    parser.add_argument("--beta", type=float, default=0.05, help="Valor inicial beta del filtro")
    parser.add_argument("--pupil-offset", type=float, default=0.35, help="Desplazamiento máximo relativo de pupila")
    return parser.parse_args()


def setup_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {index}")
    return cap


def setup_controls(initial_min_cutoff: float, initial_beta: float, initial_offset: float) -> None:
    cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("controls", 320, 120)

    def to_slider(value: float, scale: float = 100.0) -> int:
        return int(value * scale)

    cv2.createTrackbar("min_cutoff x100", "controls", to_slider(initial_min_cutoff), 500, lambda _x: None)
    cv2.createTrackbar("beta x100", "controls", to_slider(initial_beta), 500, lambda _x: None)
    cv2.createTrackbar("pupil%", "controls", int(initial_offset * 100), 100, lambda _x: None)


def read_controls(default_min_cutoff: float, default_beta: float, default_offset: float) -> tuple[float, float, float]:
    if cv2.getWindowProperty("controls", cv2.WND_PROP_VISIBLE) < 1:
        return default_min_cutoff, default_beta, default_offset

    min_cutoff = cv2.getTrackbarPos("min_cutoff x100", "controls") / 100.0
    beta = cv2.getTrackbarPos("beta x100", "controls") / 100.0
    pupil_offset = cv2.getTrackbarPos("pupil%", "controls") / 100.0
    return max(0.01, min_cutoff), max(0.0, beta), np.clip(pupil_offset, 0.1, 0.6)


def detection_to_screen(
    detection: PersonDetection,
    frame_shape: tuple[int, int, int],
    renderer_cfg: EyeRenderConfig,
    homography: Optional[np.ndarray],
) -> tuple[float, float]:
    cx, cy = detection.centroid
    if homography is not None:
        px, py = apply_homography(homography, cx, cy)
    else:
        height, width = frame_shape[:2]
        px = np.interp(cx, [0, width], [0, renderer_cfg.width])
        py = np.interp(cy, [0, height], [0, renderer_cfg.height])
    nx = (px / renderer_cfg.width) * 2.0 - 1.0
    ny = (py / renderer_cfg.height) * 2.0 - 1.0
    return float(nx), float(ny)


def main() -> None:
    args = parse_args()

    homography = None
    if args.homography:
        homography = load_homography(args.homography)

    tracker = PersonTracker(
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )

    renderer_cfg = EyeRenderConfig(
        width=args.output_width,
        height=args.output_height,
        pupil_max_offset=args.pupil_offset,
    )
    renderer = EyeRenderer(renderer_cfg)

    filter_x = OneEuroFilter(freq=args.initial_freq, min_cutoff=args.min_cutoff, beta=args.beta)
    filter_y = OneEuroFilter(freq=args.initial_freq, min_cutoff=args.min_cutoff, beta=args.beta)

    cap = setup_camera(args.camera)

    cv2.namedWindow("eyes", cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty("eyes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow("eyes", renderer_cfg.width, renderer_cfg.height)

    if args.controls:
        setup_controls(args.min_cutoff, args.beta, args.pupil_offset)

    if args.show_camera:
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    print("Presiona ESC o Q para salir, B para forzar parpadeo.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fin del stream de la cámara.")
                break

            detection = tracker.best_detection(frame)
            now = time.perf_counter()

            if detection:
                nx, ny = detection_to_screen(detection, frame.shape, renderer_cfg, homography)
            else:
                nx, ny = 0.0, 0.0

            if args.controls:
                min_cutoff, beta, pupil_offset = read_controls(args.min_cutoff, args.beta, renderer_cfg.pupil_max_offset)
                filter_x.update_params(min_cutoff=min_cutoff, beta=beta)
                filter_y.update_params(min_cutoff=min_cutoff, beta=beta)
                renderer.config.pupil_max_offset = pupil_offset

            filtered_x = filter_x(nx, timestamp=now)
            filtered_y = filter_y(ny, timestamp=now)
            renderer.update_target(filtered_x, filtered_y)

            canvas = renderer.render()
            cv2.imshow("eyes", canvas)

            if args.show_camera:
                debug_frame = frame.copy()
                if detection:
                    x1, y1, x2, y2 = map(int, detection.bbox)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(debug_frame, (int(detection.centroid[0]), int(detection.centroid[1])), 6, (0, 255, 0), -1)
                    cv2.putText(
                        debug_frame,
                        f"id={detection.track_id} conf={detection.confidence:.2f}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("camera", debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("b"), ord("B")):
                renderer.force_blink()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
