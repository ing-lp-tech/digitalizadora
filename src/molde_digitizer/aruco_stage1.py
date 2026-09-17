from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera import CameraProfile, undistort_image
from .fiducials import (
    detect_aruco_markers,
    estimate_image_to_board_mm,
    leave_one_marker_out_metrics,
    marker_correspondences,
    transform_points,
)
from .stage1 import load_image


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"No se pudo codificar {path.name}")
    encoded.tofile(path)


def _load_board_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("La configuración de pizarra debe usar schema_version 1")
    if "dictionary" not in config or "board_size_mm" not in config or "markers" not in config:
        raise ValueError("La configuración no contiene diccionario, tamaño y marcadores")
    width, height = map(float, config["board_size_mm"])
    if width <= 0 or height <= 0:
        raise ValueError("Las dimensiones físicas de la pizarra deben ser positivas")
    if len(config["markers"]) < 4:
        raise ValueError("La configuración necesita al menos cuatro marcadores")
    return config


def _spatial_metrics(
    board_points_mm: np.ndarray,
    inliers: np.ndarray,
    board_size_mm: tuple[float, float],
    image_to_board_mm: np.ndarray,
    image_size_px: tuple[int, int],
) -> dict[str, Any]:
    width_mm, height_mm = board_size_mm
    inlier_points = np.asarray(board_points_mm, np.float32)[inliers]
    hull = cv2.convexHull(inlier_points)
    hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0
    coverage = hull_area / (width_mm * height_mm)
    board_corners_mm = np.array(
        [[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]],
        np.float32,
    )
    board_corners_image = transform_points(board_corners_mm, np.linalg.inv(image_to_board_mm))
    image_width, image_height = image_size_px
    normalized_margins = np.column_stack(
        (
            board_corners_image[:, 0] / image_width,
            (image_width - 1 - board_corners_image[:, 0]) / image_width,
            board_corners_image[:, 1] / image_height,
            (image_height - 1 - board_corners_image[:, 1]) / image_height,
        )
    )
    inside = (
        (board_corners_image[:, 0] >= 0)
        & (board_corners_image[:, 0] < image_width)
        & (board_corners_image[:, 1] >= 0)
        & (board_corners_image[:, 1] < image_height)
    )
    return {
        "inlier_hull_board_coverage_ratio": round(coverage, 6),
        "projected_board_corners_image_px": np.round(board_corners_image, 4).tolist(),
        "projected_board_corners_inside_fraction": round(float(np.mean(inside)), 6),
        "projected_board_min_image_margin_normalized": round(float(np.min(normalized_margins)), 7),
    }


def _quality_gate(
    marker_count: int,
    reprojection: dict[str, Any],
    spatial: dict[str, Any],
    cross_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    if marker_count < 4:
        reasons.append("Se detectaron menos de cuatro marcadores configurados")
    if reprojection["inlier_count"] < 12:
        reasons.append("Hay menos de doce esquinas inlier")
    if reprojection["inlier_ratio"] < 0.70:
        reasons.append("La proporción de correspondencias compatibles es inferior al 70 %")
    if reprojection["inlier_error_mm"]["rms"] > 1.0:
        reasons.append("El RMS de reproyección supera 1 mm")
    if reprojection["inlier_error_mm"]["max"] > 2.0:
        reasons.append("El error máximo de reproyección supera 2 mm")
    if spatial["inlier_hull_board_coverage_ratio"] < 0.25:
        reasons.append("Las referencias inlier cubren menos del 25 % de la pizarra")
    if spatial["projected_board_corners_inside_fraction"] < 1.0:
        reasons.append("La pizarra no está completamente contenida en la fotografía")
    elif spatial["projected_board_min_image_margin_normalized"] < 0.005:
        warnings.append("La pizarra está demasiado cerca del borde de la fotografía")
    if cross_validation and cross_validation.get("available"):
        if cross_validation["p95_mm"] > 1.5:
            reasons.append("El P95 leave-one-marker-out supera 1,5 mm")
        if cross_validation["max_mm"] > 3.0:
            reasons.append("El máximo leave-one-marker-out supera 3 mm")
    elif marker_count >= 4:
        warnings.append("No se pudo completar la validación leave-one-marker-out")
    return {
        "decision": "reject" if reasons else ("warn" if warnings else "pass"),
        "rejection_reasons": reasons,
        "warnings": warnings,
        "thresholds": {
            "minimum_markers": 4,
            "minimum_inlier_corners": 12,
            "minimum_inlier_ratio": 0.70,
            "maximum_inlier_rms_mm": 1.0,
            "maximum_inlier_error_mm": 2.0,
            "minimum_inlier_board_coverage_ratio": 0.25,
            "board_must_be_fully_visible": True,
            "maximum_leave_one_marker_out_p95_mm": 1.5,
            "maximum_leave_one_marker_out_max_mm": 3.0,
        },
        "scope": "Calidad geométrica según la configuración; no certifica fabricación ni ground truth",
    }


def _overlay(
    image: np.ndarray,
    detections: list[Any],
    matched_ids: list[int],
    inliers: np.ndarray,
    board_corners_image: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    inlier_by_id = {
        marker_id: inliers[index * 4 : index * 4 + 4]
        for index, marker_id in enumerate(matched_ids)
    }
    for detection in detections:
        marker_inliers = inlier_by_id.get(detection.marker_id)
        if marker_inliers is None:
            color = (0, 165, 255)
        elif bool(np.all(marker_inliers)):
            color = (40, 220, 40)
        else:
            color = (30, 30, 230)
        corners = np.round(detection.corners_image_px).astype(np.int32)
        cv2.polylines(output, [corners], True, color, 3)
        center = tuple(np.round(detection.corners_image_px.mean(axis=0)).astype(int))
        cv2.putText(
            output,
            str(detection.marker_id),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.polylines(
        output,
        [np.round(board_corners_image).astype(np.int32)],
        True,
        (255, 100, 20),
        3,
    )
    return output


def process_aruco_stage1(
    image_path: Path,
    output_dir: Path,
    board_config_path: Path,
    camera_profile_path: Path | None = None,
    pixels_per_mm: float = 1.0,
    ransac_threshold_mm: float = 1.5,
) -> dict[str, Any]:
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm debe ser positivo")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _load_board_config(board_config_path)
    board_width_mm, board_height_mm = map(float, config["board_size_mm"])
    original = load_image(Path(image_path))
    processing = original
    camera_correction: dict[str, Any] = {
        "applied": False,
        "profile_path": None,
        "reason": "No se proporcionó un perfil de cámara",
    }
    if camera_profile_path is not None:
        profile = CameraProfile.load(camera_profile_path)
        processing, details = undistort_image(original, profile, alpha=0.0)
        camera_correction = {
            "applied": True,
            "profile_path": str(Path(camera_profile_path).resolve()),
            **details,
        }
        _write_image(output_dir / "undistorted.jpg", processing)

    detections = detect_aruco_markers(processing, str(config["dictionary"]))
    image_points, board_points_mm, matched_ids = marker_correspondences(detections, config)
    if len(matched_ids) < 2:
        raise RuntimeError(
            f"Sólo se asociaron {len(matched_ids)} marcadores; no se puede estimar una homografía robusta"
        )
    image_to_board_mm, inliers, reprojection = estimate_image_to_board_mm(
        image_points,
        board_points_mm,
        ransac_threshold_mm=ransac_threshold_mm,
    )
    spatial = _spatial_metrics(
        board_points_mm,
        inliers,
        (board_width_mm, board_height_mm),
        image_to_board_mm,
        (processing.shape[1], processing.shape[0]),
    )
    cross_validation = leave_one_marker_out_metrics(
        image_points,
        board_points_mm,
        matched_ids,
        ransac_threshold_mm=ransac_threshold_mm,
    )
    gate = _quality_gate(len(matched_ids), reprojection, spatial, cross_validation)
    board_corners_image = np.asarray(spatial["projected_board_corners_image_px"], np.float32)

    scale = np.array(
        [[pixels_per_mm, 0, 0], [0, pixels_per_mm, 0], [0, 0, 1]],
        np.float64,
    )
    image_to_rectified = scale @ image_to_board_mm
    output_size = (
        max(1, int(round(board_width_mm * pixels_per_mm))),
        max(1, int(round(board_height_mm * pixels_per_mm))),
    )
    rectified = cv2.warpPerspective(processing, image_to_rectified, output_size)

    serialized_detections = []
    matched_index = {marker_id: index for index, marker_id in enumerate(matched_ids)}
    for detection in detections:
        index = matched_index.get(detection.marker_id)
        serialized_detections.append(
            {
                "marker_id": detection.marker_id,
                "configured": index is not None,
                "corners_processing_px": np.round(detection.corners_image_px, 4).tolist(),
                "corner_inliers": inliers[index * 4 : index * 4 + 4].tolist() if index is not None else None,
            }
        )
    physical_validation = dict(config.get("physical_validation", {}))
    detections_report = {
        "schema_version": 1,
        "source_image": str(Path(image_path).resolve()),
        "board_config": str(Path(board_config_path).resolve()),
        "board_config_status": config.get("status", "unspecified"),
        "camera_correction": camera_correction,
        "units": {
            "source": "image_px",
            "processing": "undistorted_px" if camera_correction["applied"] else "image_px",
            "board": "board_mm",
            "rectified": "rectified_px",
        },
        "board_size_mm": [board_width_mm, board_height_mm],
        "pixels_per_mm": pixels_per_mm,
        "image_to_board_mm": np.round(image_to_board_mm, 12).tolist(),
        "image_to_rectified_px": np.round(image_to_rectified, 12).tolist(),
        "detections": serialized_detections,
        "matched_marker_ids": matched_ids,
    }
    quality = {
        "schema_version": 1,
        "status": "geometry_pass" if gate["decision"] == "pass" else gate["decision"],
        "capture_gate": gate,
        "reprojection": reprojection,
        "leave_one_marker_out": cross_validation,
        "spatial": spatial,
        "detected_marker_count": len(detections),
        "matched_marker_count": len(matched_ids),
        "physical_scale_available": True,
        "physical_validation": physical_validation,
        "millimeter_accuracy_validated": bool(physical_validation.get("validated", False)),
        "limitations": [
            "Un gate geométrico aprobado no valida por sí solo la fabricación de la pizarra.",
            "La exactitud comercial exige comparar contra puntos físicos independientes.",
        ],
    }
    (output_dir / "detections.json").write_text(
        json.dumps(detections_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_image(
        output_dir / "references_overlay.jpg",
        _overlay(processing, detections, matched_ids, inliers, board_corners_image),
    )
    _write_image(output_dir / "rectified.jpg", rectified)
    return {
        "image": Path(image_path).name,
        "decision": gate["decision"],
        "detected_marker_count": len(detections),
        "matched_marker_count": len(matched_ids),
        "inlier_corner_count": reprojection["inlier_count"],
        "inlier_rms_mm": reprojection["inlier_error_mm"]["rms"],
        "inlier_max_mm": reprojection["inlier_error_mm"]["max"],
        "holdout_p95_mm": cross_validation.get("p95_mm")
        if cross_validation.get("available")
        else None,
        "board_coverage_ratio": spatial["inlier_hull_board_coverage_ratio"],
        "output_dir": str(output_dir),
    }
