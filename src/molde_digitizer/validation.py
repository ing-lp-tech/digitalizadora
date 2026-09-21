from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _error_statistics(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, np.float64).ravel()
    return {
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def validate_ground_truth_data(
    detections: dict[str, Any],
    observations: dict[str, Any],
    maximum_p95_mm: float = 2.0,
    maximum_error_mm: float = 3.0,
) -> dict[str, Any]:
    expected_space = detections["units"]["processing"]
    observed_space = observations.get("coordinate_space")
    if observed_space != expected_space:
        raise ValueError(
            f"Los puntos están en {observed_space!r}, pero la homografía espera {expected_space!r}"
        )
    points = observations.get("points", [])
    if len(points) < 5:
        raise ValueError("Se requieren al menos cinco puntos físicos independientes")
    observed_px = np.asarray([point["observed_processing_px"] for point in points], np.float32)
    expected_mm = np.asarray([point["expected_board_mm"] for point in points], np.float32)
    if observed_px.shape != expected_mm.shape or observed_px.shape[1:] != (2,):
        raise ValueError("Cada observación debe tener dos coordenadas observadas y dos esperadas")
    homography = np.asarray(detections["image_to_board_mm"], np.float64)
    recovered_mm = cv2.perspectiveTransform(observed_px.reshape(1, -1, 2), homography)[0]
    errors = np.linalg.norm(recovered_mm - expected_mm, axis=1)
    point_results = []
    for source, expected, recovered, error in zip(points, expected_mm, recovered_mm, errors):
        point_results.append(
            {
                "name": source.get("name"),
                "expected_board_mm": expected.tolist(),
                "recovered_board_mm": recovered.tolist(),
                "error_mm": float(error),
            }
        )

    distances = []
    point_by_name = {point.get("name"): index for index, point in enumerate(points)}
    for measurement in observations.get("distances", []):
        first_name = measurement["point_a"]
        second_name = measurement["point_b"]
        if first_name not in point_by_name or second_name not in point_by_name:
            raise ValueError(f"Distancia con punto desconocido: {first_name} / {second_name}")
        first = recovered_mm[point_by_name[first_name]]
        second = recovered_mm[point_by_name[second_name]]
        recovered_distance = float(np.linalg.norm(second - first))
        expected_distance = float(measurement["expected_mm"])
        distances.append(
            {
                "name": measurement.get("name"),
                "expected_mm": expected_distance,
                "recovered_mm": recovered_distance,
                "absolute_error_mm": abs(recovered_distance - expected_distance),
            }
        )

    stats = _error_statistics(errors)
    reasons = []
    if stats["p95"] > maximum_p95_mm:
        reasons.append(f"P95 {stats['p95']:.3f} mm supera {maximum_p95_mm:.3f} mm")
    if stats["max"] > maximum_error_mm:
        reasons.append(f"Máximo {stats['max']:.3f} mm supera {maximum_error_mm:.3f} mm")
    report = {
        "schema_version": 1,
        "coordinate_space": expected_space,
        "point_count": len(points),
        "point_error_mm": stats,
        "points": point_results,
        "distances": distances,
        "quality_gate": {
            "decision": "pass" if not reasons else "reject",
            "rejection_reasons": reasons,
            "maximum_p95_mm": maximum_p95_mm,
            "maximum_error_mm": maximum_error_mm,
        },
        "independence_statement": observations.get("independence_statement"),
        "measurement_provenance": observations.get("measurement_provenance"),
    }
    return report


def validate_ground_truth_files(
    detections_path: Path,
    observations_path: Path,
    output_path: Path,
    maximum_p95_mm: float = 2.0,
    maximum_error_mm: float = 3.0,
) -> dict[str, Any]:
    detections = json.loads(Path(detections_path).read_text(encoding="utf-8"))
    observations = json.loads(Path(observations_path).read_text(encoding="utf-8"))
    report = validate_ground_truth_data(
        detections,
        observations,
        maximum_p95_mm=maximum_p95_mm,
        maximum_error_mm=maximum_error_mm,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

