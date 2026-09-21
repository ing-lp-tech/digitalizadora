"""Mide tiempo y memoria de process_marker_metric sobre una foto real.

Referencia local (esta maquina), para comparar contra la medicion en el
servidor de Render y saber cuanto es overhead del free tier vs. costo real
del procesamiento.

Uso:
    python scripts/benchmark_motor.py [ruta_foto] [--repeticiones N]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molde_digitizer.marker_metric import process_marker_metric  # noqa: E402

try:
    import psutil

    _PROCESO = psutil.Process()
except ImportError:  # psutil es opcional, solo para este benchmark
    _PROCESO = None


def _rss_mb() -> float | None:
    if _PROCESO is None:
        return None
    return _PROCESO.memory_info().rss / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "foto",
        nargs="?",
        default=str(ROOT / "datos_ejemplo_fotos_aruco" / "20260909_184126.jpg"),
    )
    parser.add_argument("--repeticiones", type=int, default=5)
    args = parser.parse_args()

    output_dir = ROOT / "resultados" / "_benchmark"
    tiempos = []

    rss_antes = _rss_mb()
    tracemalloc.start()

    for i in range(args.repeticiones):
        shutil.rmtree(output_dir, ignore_errors=True)
        inicio = time.perf_counter()
        process_marker_metric(args.foto, output_dir=output_dir, marker_size_mm=50, px_per_mm=2)
        tiempos.append(time.perf_counter() - inicio)

    _, pico_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_despues = _rss_mb()

    shutil.rmtree(output_dir, ignore_errors=True)

    print(f"Foto: {args.foto}")
    print(f"Repeticiones: {args.repeticiones}")
    print(f"Tiempo por foto: min={min(tiempos):.2f}s  max={max(tiempos):.2f}s  "
          f"promedio={sum(tiempos)/len(tiempos):.2f}s")
    print(f"Pico memoria Python (tracemalloc): {pico_tracemalloc / (1024*1024):.1f} MB")
    if _PROCESO is not None:
        print(f"RSS del proceso: antes={rss_antes:.1f} MB  despues={rss_despues:.1f} MB "
              f"(incluye interprete + OpenCV, no solo Python puro)")
    else:
        print("psutil no instalado: sin medicion de RSS (solo tracemalloc, que no ve "
              "buffers nativos de OpenCV/numpy).")


if __name__ == "__main__":
    main()
