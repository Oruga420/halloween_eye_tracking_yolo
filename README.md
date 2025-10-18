# Luna Eyes Tracker

Sistema interactivo para proyectar ojos que siguen a la persona más prominente en la escena usando visión por computadora.

## Requisitos

- Python 3.11+
- Dependencias: `pip install -r requirements.txt`
- Modelo YOLO compatible (por defecto `yolo11n.pt` de Ultralytics; se descarga automáticamente la primera vez).

## Estructura

- `luna_eyes/` – módulos reutilizables (tracking, filtros, render, calibración).
- `scripts/calibrate.py` – asistente de homografía con cuatro puntos.
- `scripts/run_eyes.py` – loop principal: captura, tracking y render de ojos.
- `data/homography.npy` – matriz opcional guardada con el script de calibración.

## Uso rápido

1. Calibrar una vez con el proyector encendido:
   ```bash
   python scripts/calibrate.py --camera 0 --output data/homography.npy
   ```
2. Ejecutar la instalación:
   ```bash
   python scripts/run_eyes.py --camera 0 --homography data/homography.npy --fullscreen
   ```

Configura los sliders en la ventana de debug para ajustar suavizado, límites y velocidad de los ojos en tiempo real.
