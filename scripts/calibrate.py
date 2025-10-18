"""
Asistente interactivo para capturar homografía cámara → proyector.
Instrucciones:
1. Proyecta una cuadrícula o recuadro en la cortina.
2. Haz clic en el stream de la cámara en el orden: esquina superior izquierda,
   superior derecha, inferior derecha, inferior izquierda.
3. Presiona ENTER para guardar cuando las 4 esquinas estén marcadas.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from luna_eyes.calibration import compute_homography, save_homography

Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibración de homografía para Luna Eyes.")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default: 0)")
    parser.add_argument("--output", type=str, required=True, help="Ruta de salida .npy para guardar la homografía")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de la proyección")
    parser.add_argument("--height", type=int, default=720, help="Alto de la proyección")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.camera}")

    win_name = "calibration"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)
    picked_points: List[Point] = []

    def on_click(event, x, y, _flags, _userdata):
        nonlocal picked_points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(picked_points) >= 4:
                return
            picked_points.append((float(x), float(y)))
            print(f"Punto {len(picked_points)} = ({x}, {y})")

    cv2.setMouseCallback(win_name, on_click)

    print("Haz clic en las 4 esquinas en orden (sup izq, sup der, inf der, inf izq).")
    print("Presiona R para reiniciar, ENTER para guardar, ESC para salir sin cambios.")

    H = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Fin del stream de la cámara.")
                break

            display = frame.copy()
            for idx, point in enumerate(picked_points):
                color = (0, 255, 0) if idx < 4 else (0, 0, 255)
                cv2.circle(display, (int(point[0]), int(point[1])), 6, color, -1)
                cv2.putText(
                    display,
                    f"{idx+1}",
                    (int(point[0]) + 10, int(point[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if len(picked_points) == 4:
                dst = [
                    (0.0, 0.0),
                    (float(args.width), 0.0),
                    (float(args.width), float(args.height)),
                    (0.0, float(args.height)),
                ]
                try:
                    H = compute_homography(picked_points, dst)
                except Exception as error:  # pragma: no cover - feedback visual
                    print(f"Error calculando homografía: {error}")
                    H = None
                else:
                    # Dibuja el polígono proyectado como referencia
                    src_rect = np.array(
                        [[(0, 0), (args.width, 0), (args.width, args.height), (0, args.height)]],
                        dtype=np.float32,
                    )
                    invH = np.linalg.inv(H)
                    mapped_rect = cv2.perspectiveTransform(src_rect, invH)[0]
                    rect_pts = np.int32(mapped_rect)
                    cv2.polylines(display, [rect_pts], True, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                H = None
                break
            if key in (13, 10) and H is not None:
                save_homography(Path(args.output), H)
                print(f"Homografía guardada en {args.output}")
                break
            if key in (ord("r"), ord("R")):
                picked_points.clear()
                H = None
                print("Puntos reiniciados.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
