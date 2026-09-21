# digitalizadora — detalle técnico

Motor local de visión por computadora para digitalizar moldes de indumentaria
a partir de fotos de celular tomadas sobre una pizarra con marcadores ArUco de
referencia: rectifica el plano a milímetros y segmenta los contornos de los
moldes.

Este repo contiene el motor de digitalización (hito 2 del presupuesto
acordado: "motor de digitalización funcionando, primeros resultados
validables"). No incluye la aplicación web de edición ni la exportación a
DXF/SVG, que corresponden al hito 3 del presupuesto.

- `src/molde_digitizer/` — detección de marcadores ArUco, rectificación
  métrica del plano y segmentación de moldes.
- `datos_ejemplo_fotos_aruco/` — fotos reales de la pizarra con los 14
  marcadores ArUco pegados, para probar el motor.

## Stack tecnológico

- Python 3.11
- `numpy`, `opencv-contrib-python`, `Pillow` (ver `requirements.txt`)
- Sin framework, sin base de datos ni servicios externos.

## Variables de entorno

Ninguna. Corre 100% local.

## Instalación y uso

```powershell
pip install -r requirements.txt
```

```python
import sys
sys.path.insert(0, "src")
from molde_digitizer.marker_metric import process_marker_metric

process_marker_metric(
    "datos_ejemplo_fotos_aruco/20260909_184126.jpg",
    output_dir="resultados/prueba",
    marker_size_mm=50,
    px_per_mm=2,
)
```

Genera en `output_dir` un reporte JSON con los contornos de cada molde en
`contour_mm` (coordenadas ya en milímetros) más imágenes de depuración.

## Hosting

No aplica: es una librería/motor, no un servicio desplegado.

## CI/CD

No hay pipeline configurado en esta etapa; se agregará cuando el proyecto
incluya la aplicación completa (hito 3).
