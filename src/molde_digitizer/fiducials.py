from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .stage1 import reprojection_metrics


@dataclass(frozen=True)
class MarkerDetection:
    marker_id: int
    corners_image_px: np.ndarray


def aruco_dictionary(name: str) -> cv2.aruco.Dictionary:
    if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        raise ValueError(f"Diccionario ArUco desconocido: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_aruco_markers(image: np.ndarray, dictionary_name: str) -> list[MarkerDetection]:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dictionary(dictionary_name), parameters)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return []
    detections = [
        MarkerDetection(int(marker_id), np.asarray(marker_corners, np.float32).reshape(4, 2))
        for marker_corners, marker_id in zip(corners, ids.ravel())
    ]
    detections.sort(key=lambda detection: detection.marker_id)
    return detections


def marker_correspondences(
    detections: list[MarkerDetection], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    configured = config.get("markers", {})
    image_points: list[np.ndarray] = []
    physical_points: list[np.ndarray] = []
    matched_ids: list[int] = []
    used_ids: set[int] = set()
    for detection in detections:
        key = str(detection.marker_id)
        if key not in configured or detection.marker_id in used_ids:
            continue
        physical = np.asarray(configured[key]["corners_board_mm"], dtype=np.float32)
        if physical.shape != (4, 2):
            raise ValueError(f"El marcador {key} no tiene cuatro esquinas físicas")
        image_points.append(detection.corners_image_px)
        physical_points.append(physical)
        matched_ids.append(detection.marker_id)
        used_ids.add(detection.marker_id)
    if not image_points:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            [],
        )
    return np.vstack(image_points), np.vstack(physical_points), matched_ids


def estimate_image_to_board_mm(
    image_points: np.ndarray,
    board_points_mm: np.ndarray,
    ransac_threshold_mm: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = np.asarray(image_points, np.float32).reshape(-1, 2)
    destination = np.asarray(board_points_mm, np.float32).reshape(-1, 2)
    if len(source) < 8 or len(source) != len(destination):
        raise ValueError("Se requieren al menos dos marcadores completos y correspondencias parejas")
    homography, inlier_mask = cv2.findHomography(
        source,
        destination,
        cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_mm),
        maxIters=5000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("RANSAC no pudo estimar una homografía válida")
    inliers = inlier_mask.ravel().astype(bool)
    if int(inliers.sum()) < 8:
        raise RuntimeError("La homografía tiene menos de ocho esquinas inlier")
    all_metrics = reprojection_metrics(source, destination, homography)
    inlier_metrics = reprojection_metrics(source[inliers], destination[inliers], homography)
    quality = {
        "point_count": int(len(source)),
        "inlier_count": int(inliers.sum()),
        "outlier_count": int((~inliers).sum()),
        "inlier_ratio": round(float(np.mean(inliers)), 6),
        "ransac_threshold_mm": float(ransac_threshold_mm),
        "inlier_error_mm": {key: round(value, 6) for key, value in inlier_metrics.items()},
        "all_point_error_mm": {key: round(value, 6) for key, value in all_metrics.items()},
    }
    return homography, inliers, quality


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, np.float32).reshape(1, -1, 2), homography
    )[0]


def leave_one_marker_out_metrics(
    image_points: np.ndarray,
    board_points_mm: np.ndarray,
    marker_ids: list[int],
    ransac_threshold_mm: float = 1.5,
) -> dict[str, Any]:
    source = np.asarray(image_points, np.float32).reshape(-1, 2)
    destination = np.asarray(board_points_mm, np.float32).reshape(-1, 2)
    if len(source) != len(marker_ids) * 4 or len(destination) != len(source):
        raise ValueError("Cada marcador debe aportar exactamente cuatro correspondencias")
    if len(marker_ids) < 4:
        return {
            "available": False,
            "reason": "Se necesitan al menos cuatro marcadores para validación cruzada",
            "marker_count": len(marker_ids),
        }
    per_marker = []
    all_errors: list[float] = []
    for marker_index, marker_id in enumerate(marker_ids):
        held_slice = slice(marker_index * 4, marker_index * 4 + 4)
        train_mask = np.ones(len(source), dtype=bool)
        train_mask[held_slice] = False
        homography, inlier_mask = cv2.findHomography(
            source[train_mask],
            destination[train_mask],
            cv2.RANSAC,
            ransacReprojThreshold=float(ransac_threshold_mm),
            maxIters=5000,
            confidence=0.999,
        )
        if homography is None or inlier_mask is None or int(inlier_mask.sum()) < 8:
            per_marker.append({"marker_id": marker_id, "status": "homography_failed"})
            continue
        projected = transform_points(source[held_slice], homography)
        errors = np.linalg.norm(projected - destination[held_slice], axis=1)
        all_errors.extend(map(float, errors))
        per_marker.append(
            {
                "marker_id": marker_id,
                "status": "measured",
                "training_inlier_count": int(inlier_mask.sum()),
                "corner_errors_mm": errors.tolist(),
                "rms_mm": float(np.sqrt(np.mean(np.square(errors)))),
                "max_mm": float(np.max(errors)),
            }
        )
    if len(all_errors) < 16:
        return {
            "available": False,
            "reason": "No se pudieron medir suficientes marcadores reservados",
            "marker_count": len(marker_ids),
            "per_marker": per_marker,
        }
    errors_array = np.asarray(all_errors, np.float64)
    return {
        "available": True,
        "marker_count": len(marker_ids),
        "corner_count": int(len(errors_array)),
        "rms_mm": float(np.sqrt(np.mean(np.square(errors_array)))),
        "median_mm": float(np.median(errors_array)),
        "p95_mm": float(np.percentile(errors_array, 95)),
        "max_mm": float(np.max(errors_array)),
        "per_marker": per_marker,
        "note": "Cada marcador se evalúa con una homografía estimada sin sus cuatro esquinas.",
    }
