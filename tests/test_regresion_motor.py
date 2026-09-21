"""Test de regresión: corre el motor sobre una foto real conocida y verifica
que las medidas no se corran de los valores observados al momento de esta
entrega. No certifica exactitud contra el molde físico (eso lo hace
scripts/validar_ground_truth.py en el repo completo) — solo detecta si un
cambio de código altera el resultado sin que nadie se dé cuenta.
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molde_digitizer.marker_metric import process_marker_metric  # noqa: E402

FOTO = ROOT / "datos_ejemplo_fotos_aruco" / "20260909_184126.jpg"

# Baseline observado corriendo el motor sobre FOTO con marker_size_mm=50, px_per_mm=2.
# Tolerancias generosas (no son el gate de precisión del hito 4, solo detectan
# que el resultado se movió).
MOLDES_ESPERADOS_MM = {
    1: {"width_mm": 132.0, "height_mm": 257.0},
    2: {"width_mm": 196.0, "height_mm": 145.5},
    3: {"width_mm": 171.0, "height_mm": 177.5},
}
TOLERANCIA_MM = 2.0


class RegresionMotorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = ROOT / "resultados" / "_test_regresion"
        self.addCleanup(shutil.rmtree, self.output_dir, ignore_errors=True)

    def test_medidas_no_se_corren_sobre_foto_conocida(self) -> None:
        self.assertTrue(FOTO.exists(), f"falta la foto de referencia: {FOTO}")

        resultado = process_marker_metric(
            FOTO,
            output_dir=self.output_dir,
            marker_size_mm=50,
            px_per_mm=2,
        )

        self.assertEqual(resultado["decision"], "pass")
        self.assertEqual(resultado["used_marker_count"], 14)
        self.assertEqual(resultado["mould_count"], len(MOLDES_ESPERADOS_MM))

        moldes_por_id = {m["id"]: m for m in resultado["moulds_mm"]}
        for mould_id, esperado in MOLDES_ESPERADOS_MM.items():
            self.assertIn(mould_id, moldes_por_id)
            medido = moldes_por_id[mould_id]
            self.assertAlmostEqual(
                medido["width_mm"], esperado["width_mm"], delta=TOLERANCIA_MM,
                msg=f"molde {mould_id}: ancho se corrio del baseline",
            )
            self.assertAlmostEqual(
                medido["height_mm"], esperado["height_mm"], delta=TOLERANCIA_MM,
                msg=f"molde {mould_id}: alto se corrio del baseline",
            )


if __name__ == "__main__":
    unittest.main()
