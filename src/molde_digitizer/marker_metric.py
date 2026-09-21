"""Rectificación métrica del plano usando marcadores ArUco como regla física.

A diferencia de :mod:`molde_digitizer.aruco_stage1`, este módulo **no** necesita
un JSON con las coordenadas centro a centro de cada marcador en la pizarra. Sólo
asume que:

- todos los marcadores son cuadrados del mismo tamaño físico (``marker_size_mm``);
- están pegados sobre un plano;
- comparten orientación (la esquina 0 de cada marcador apunta al mismo lado,
  como pide la guía de impresión: "todas las flechas hacia arriba").

Con eso alcanza para quitar la perspectiva y fijar la escala en milímetros a
partir del lado conocido de los marcadores. El origen y la orientación del
sistema de coordenadas resultante son arbitrarios (esquina superior izquierda
del rectángulo que envuelve a los marcadores).

La rectificación se puede validar sin pieza patrón midiendo cuánto se desvía cada
marcador de un cuadrado perfecto de ``marker_size_mm`` después de rectificar
(``self_consistency``). Ese control NO reemplaza la comparación contra una medida
física independiente: sólo dice si la geometría interna es coherente.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .fiducials import detect_aruco_markers
from .stage1 import load_image

# Orden de esquinas que devuelve cv2.aruco para cada marcador, en su propio
# sistema: 0 sup-izq, 1 sup-der, 2 inf-der, 3 inf-izq.
_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"No se pudo codificar {path.name}")
    encoded.tofile(path)


def _normalization(points: np.ndarray) -> np.ndarray:
    """Similaridad de Hartley: centra en el origen y escala a distancia media sqrt(2)."""
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = float(np.sqrt(np.sum(shifted**2, axis=1)).mean())
    if mean_distance < 1e-12:
        raise RuntimeError("Los puntos de los marcadores están degenerados")
    scale = np.sqrt(2.0) / mean_distance
    return np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, np.float64).reshape(1, -1, 2), homography
    )[0]


def _line_through(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.cross(
        np.array([first[0], first[1], 1.0]), np.array([second[0], second[1], 1.0])
    )


def _null_vector(rows: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    return vh[-1]


def _best_fit_square_residual(
    corners_mm: np.ndarray, marker_size_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mejor cuadrado axis-aligned de lado fijo: sólo se ajusta la traslación."""
    template = _UNIT_SQUARE * marker_size_mm
    origin = (corners_mm - template).mean(axis=0)
    predicted = origin + template
    return predicted, corners_mm - predicted


def metric_rectification_from_corners(
    corners_by_id: dict[int, np.ndarray],
    marker_size_mm: float = 50.0,
) -> dict[str, Any]:
    """Núcleo geométrico: recibe esquinas en píxeles por ID, devuelve la homografía
    imagen -> plano en milímetros (origen/orientación arbitrarios) y métricas."""
    if len(corners_by_id) < 4:
        raise RuntimeError(
            f"Se necesitan al menos 4 marcadores coplanares; hay {len(corners_by_id)}"
        )
    marker_ids = sorted(corners_by_id)
    image_corners = np.vstack([np.asarray(corners_by_id[i], np.float64) for i in marker_ids])

    norm = _normalization(image_corners)
    normalized = {
        marker_id: _apply_homography(norm, np.asarray(corners_by_id[marker_id], np.float64))
        for marker_id in marker_ids
    }

    # Familias de rectas paralelas en el plano: bordes "x" (0->1, 3->2) y "y" (0->3, 1->2).
    x_lines: list[np.ndarray] = []
    y_lines: list[np.ndarray] = []
    for marker_id in marker_ids:
        c = normalized[marker_id]
        x_lines.append(_line_through(c[0], c[1]))
        x_lines.append(_line_through(c[3], c[2]))
        y_lines.append(_line_through(c[0], c[3]))
        y_lines.append(_line_through(c[1], c[2]))
    vanishing_x = _null_vector(np.asarray(x_lines))
    vanishing_y = _null_vector(np.asarray(y_lines))

    # Recta del infinito y rectificación afín (deja las paralelas paralelas otra vez).
    line_infinity = np.cross(vanishing_x, vanishing_y)
    if abs(line_infinity[2]) < 1e-12:
        raise RuntimeError("No se pudo estimar la recta del infinito del plano")
    line_infinity = line_infinity / line_infinity[2]
    affine = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], list(line_infinity)], dtype=np.float64
    )

    affine_corners = {
        marker_id: _apply_homography(affine, normalized[marker_id]) for marker_id in marker_ids
    }
    # En el espacio afín los marcadores son paralelogramos iguales salvo transformación
    # lineal. Se usa que son cuadrados de lado conocido para recuperar la métrica.
    edge_x: list[np.ndarray] = []
    edge_y: list[np.ndarray] = []
    for marker_id in marker_ids:
        c = affine_corners[marker_id]
        edge_x.append(c[1] - c[0])
        edge_x.append(c[2] - c[3])
        edge_y.append(c[3] - c[0])
        edge_y.append(c[2] - c[1])
    mean_edge_x = np.mean(edge_x, axis=0)
    mean_edge_y = np.mean(edge_y, axis=0)
    basis = np.column_stack([mean_edge_x, mean_edge_y])
    if abs(np.linalg.det(basis)) < 1e-12:
        raise RuntimeError("Los bordes de los marcadores son colineales tras rectificar")
    target = np.array(
        [[marker_size_mm, 0.0], [0.0, marker_size_mm]], dtype=np.float64
    )
    linear = target @ np.linalg.inv(basis)
    mirrored = bool(np.linalg.det(linear) < 0)
    if mirrored:
        linear = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64) @ linear
    metric_linear = np.array(
        [[linear[0, 0], linear[0, 1], 0.0], [linear[1, 0], linear[1, 1], 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # imagen (normalizada) -> mm, y luego imagen original -> mm.
    homography_norm_to_mm = metric_linear @ affine
    homography = homography_norm_to_mm @ norm

    # Traslada el origen a la esquina sup-izq del bounding box de los marcadores.
    all_mm = _apply_homography(homography, image_corners)
    offset = all_mm.min(axis=0)
    translation = np.array(
        [[1.0, 0.0, -offset[0]], [0.0, 1.0, -offset[1]], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    homography = translation @ homography

    # Métricas de autoconsistencia: desviación de cada marcador respecto de un
    # cuadrado perfecto de marker_size_mm tras rectificar.
    residuals: list[float] = []
    per_marker: list[dict[str, Any]] = []
    recovered_sides: list[float] = []
    for marker_id in marker_ids:
        corners_mm = _apply_homography(
            homography, np.asarray(corners_by_id[marker_id], np.float64)
        )
        _, residual = _best_fit_square_residual(corners_mm, marker_size_mm)
        point_errors = np.sqrt((residual**2).sum(axis=1))
        residuals.extend(point_errors.tolist())
        sides = [
            float(np.linalg.norm(corners_mm[1] - corners_mm[0])),
            float(np.linalg.norm(corners_mm[2] - corners_mm[1])),
            float(np.linalg.norm(corners_mm[3] - corners_mm[2])),
            float(np.linalg.norm(corners_mm[0] - corners_mm[3])),
        ]
        recovered_sides.extend(sides)
        per_marker.append(
            {
                "marker_id": marker_id,
                "corners_mm": np.round(corners_mm, 4).tolist(),
                "recovered_side_mm": round(float(np.mean(sides)), 4),
                "corner_rms_mm": round(float(np.sqrt(np.mean(point_errors**2))), 5),
                "corner_max_mm": round(float(point_errors.max()), 5),
            }
        )

    residuals_array = np.asarray(residuals, dtype=np.float64)
    sides_array = np.asarray(recovered_sides, dtype=np.float64)
    hull = cv2.convexHull(np.asarray(all_mm, dtype=np.float32))
    hull_area_mm2 = float(cv2.contourArea(hull))
    span_mm = (all_mm.max(axis=0) - all_mm.min(axis=0)).tolist()

    self_consistency = {
        "marker_count": len(marker_ids),
        "corner_rms_mm": round(float(np.sqrt(np.mean(residuals_array**2))), 5),
        "corner_median_mm": round(float(np.median(residuals_array)), 5),
        "corner_p95_mm": round(float(np.percentile(residuals_array, 95)), 5),
        "corner_max_mm": round(float(residuals_array.max()), 5),
        "recovered_side_mm": {
            "nominal": marker_size_mm,
            "mean": round(float(sides_array.mean()), 4),
            "std": round(float(sides_array.std()), 4),
            "min": round(float(sides_array.min()), 4),
            "max": round(float(sides_array.max()), 4),
        },
        "marker_hull_span_mm": [round(span_mm[0], 2), round(span_mm[1], 2)],
        "marker_hull_area_mm2": round(hull_area_mm2, 1),
        "coordinate_frame": "arbitrary_top_left_of_marker_bounding_box_x_right_y_down",
        "mirror_correction_applied": mirrored,
    }
    return {
        "homography_image_to_mm": homography,
        "marker_ids": marker_ids,
        "self_consistency": self_consistency,
        "per_marker": per_marker,
    }


def _self_consistency_gate(
    self_consistency: dict[str, Any],
    max_corner_rms_mm: float,
    max_corner_p95_mm: float,
    max_side_error_mm: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    warnings: list[str] = []
    if self_consistency["marker_count"] < 4:
        reasons.append("Menos de 4 marcadores coplanares")
    elif self_consistency["marker_count"] < 8:
        warnings.append("Menos de 8 marcadores: la rectificación es más sensible al ruido")
    if self_consistency["corner_rms_mm"] > max_corner_rms_mm:
        reasons.append(
            f"RMS de esquinas {self_consistency['corner_rms_mm']} mm supera {max_corner_rms_mm} mm"
        )
    if self_consistency["corner_p95_mm"] > max_corner_p95_mm:
        reasons.append(
            f"P95 de esquinas {self_consistency['corner_p95_mm']} mm supera {max_corner_p95_mm} mm"
        )
    side = self_consistency["recovered_side_mm"]
    worst_side_error = max(
        abs(side["max"] - side["nominal"]), abs(side["min"] - side["nominal"])
    )
    if worst_side_error > max_side_error_mm:
        reasons.append(
            f"Un lado de marcador quedó a {round(worst_side_error, 2)} mm del nominal "
            f"(límite {max_side_error_mm} mm): revisar escala de impresión o planaridad"
        )
    return {
        "decision": "reject" if reasons else ("warn" if warnings else "pass"),
        "rejection_reasons": reasons,
        "warnings": warnings,
        "thresholds": {
            "max_corner_rms_mm": max_corner_rms_mm,
            "max_corner_p95_mm": max_corner_p95_mm,
            "max_side_error_mm": max_side_error_mm,
            "min_markers": 4,
        },
        "scope": (
            "Coherencia geométrica interna de la rectificación. NO certifica milímetros "
            "reales: falta comparar contra una medida física independiente."
        ),
    }


def _flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Corrige el gradiente/reflejo dividiendo por un fondo muy suavizado.

    El fondo se estima en baja resolución (blur gaussiano sobre una versión reducida)
    para que la estimación no siga a los moldes, que pueden ser grandes; una apertura
    morfológica de kernel razonable los preservaría y no quitaría nada.
    """
    height, width = gray.shape
    scale = 8
    small = cv2.resize(
        gray, (max(1, width // scale), max(1, height // scale)), interpolation=cv2.INTER_AREA
    )
    blur_size = max(3, (min(small.shape) // 2) | 1)
    background_small = cv2.GaussianBlur(small, (blur_size, blur_size), 0)
    background = cv2.resize(background_small, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, background, scale=160)


def _interior_from_markers(
    per_marker: list[dict[str, Any]], marker_size_mm: float, px_per_mm: float, shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    corners = np.vstack([np.asarray(m["corners_mm"], np.float64) for m in per_marker])
    clearance = marker_size_mm * 1.5
    left = (corners[:, 0].min() + clearance) * px_per_mm
    right = (corners[:, 0].max() - clearance) * px_per_mm
    top = (corners[:, 1].min() + clearance) * px_per_mm
    bottom = (corners[:, 1].max() - clearance) * px_per_mm
    left = int(np.clip(round(left), 0, width - 2))
    right = int(np.clip(round(right), left + 1, width))
    top = int(np.clip(round(top), 0, height - 2))
    bottom = int(np.clip(round(bottom), top + 1, height))
    return left, top, right, bottom


def _segment_moulds(
    rectified: np.ndarray,
    interior: tuple[int, int, int, int],
    px_per_mm: float,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    left, top, right, bottom = interior
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    roi = gray[top:bottom, left:right]
    if roi.size == 0:
        raise RuntimeError("El interior útil quedó vacío")
    flattened = _flatten_illumination(cv2.GaussianBlur(roi, (5, 5), 0))
    otsu_threshold, mask = cv2.threshold(
        flattened, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    short_side = min(roi.shape)
    open_size = max(3, int(round(short_side * 0.004)) | 1)
    close_size = max(3, int(round(short_side * 0.006)) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = float(roi.shape[0] * roi.shape[1])
    minimum_area = max(1200.0, roi_area * 0.002)
    maximum_area = roi_area * 0.45
    accepted: list[np.ndarray] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not minimum_area <= area <= maximum_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 18 or h < 18 or area / float(w * h) < 0.12:
            continue
        shifted = contour.copy()
        shifted[:, 0, 0] += left
        shifted[:, 0, 1] += top
        accepted.append(shifted)

    def centroid(contour: np.ndarray) -> tuple[float, float]:
        moments = cv2.moments(contour)
        if moments["m00"]:
            return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2.0, y + h / 2.0

    accepted.sort(key=centroid)
    full_mask = np.zeros(gray.shape, dtype=np.uint8)
    shapes: list[dict[str, Any]] = []
    for index, contour in enumerate(accepted, start=1):
        cv2.drawContours(full_mask, [contour], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = centroid(contour)
        perimeter = float(cv2.arcLength(contour, True))
        simplified = cv2.approxPolyDP(contour, max(1.5, perimeter * 0.0025), True)
        shapes.append(
            {
                "id": index,
                "label": f"Molde {index}",
                "bounding_box_mm": [
                    round(x / px_per_mm, 2),
                    round(y / px_per_mm, 2),
                    round(w / px_per_mm, 2),
                    round(h / px_per_mm, 2),
                ],
                "width_mm": round(w / px_per_mm, 2),
                "height_mm": round(h / px_per_mm, 2),
                "area_mm2": round(float(cv2.contourArea(contour)) / (px_per_mm**2), 1),
                "perimeter_mm": round(perimeter / px_per_mm, 2),
                "centroid_mm": [round(center_x / px_per_mm, 2), round(center_y / px_per_mm, 2)],
                "contour_px": simplified[:, 0, :].astype(int).tolist(),
            }
        )
    metadata = {
        "otsu_threshold_flattened": round(float(otsu_threshold), 2),
        "interior_bounds_px": [left, top, right, bottom],
        "minimum_area_px2": round(minimum_area, 1),
        "px_per_mm": px_per_mm,
        "illumination_flattened": True,
    }
    return full_mask, shapes, metadata


def _markers_overlay(image: np.ndarray, detections: list[Any], homography: np.ndarray) -> np.ndarray:
    output = image.copy()
    thickness = max(2, min(image.shape[:2]) // 600)
    for detection in detections:
        corners = np.round(detection.corners_image_px).astype(np.int32)
        cv2.polylines(output, [corners], True, (40, 220, 40), thickness)
        center = tuple(np.round(detection.corners_image_px.mean(axis=0)).astype(int))
        cv2.putText(
            output, str(detection.marker_id), center, cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 240, 255), thickness, cv2.LINE_AA,
        )
    return output


def _shapes_overlay(rectified: np.ndarray, shapes: list[dict[str, Any]]) -> np.ndarray:
    palette = (
        (79, 214, 255), (110, 231, 183), (252, 165, 165), (196, 181, 253),
        (147, 197, 253), (253, 224, 71), (251, 146, 60), (244, 114, 182),
    )
    overlay = rectified.copy()
    tint = rectified.copy()
    for shape in shapes:
        contour = np.asarray(shape["contour_px"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(tint, [contour], -1, palette[(shape["id"] - 1) % len(palette)], cv2.FILLED)
    overlay = cv2.addWeighted(tint, 0.25, overlay, 0.75, 0)
    for shape in shapes:
        contour = np.asarray(shape["contour_px"], dtype=np.int32).reshape(-1, 1, 2)
        color = palette[(shape["id"] - 1) % len(palette)]
        cv2.drawContours(overlay, [contour], -1, color, 3, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(
            overlay, f"{shape['id']}: {shape['width_mm']:.0f}x{shape['height_mm']:.0f} mm",
            (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
    return overlay


def process_marker_metric(
    image_path: Path,
    output_dir: Path,
    dictionary: str = "DICT_5X5_100",
    marker_size_mm: float = 50.0,
    px_per_mm: float = 2.0,
    scale_correction: float = 1.0,
    max_corner_rms_mm: float = 1.5,
    max_corner_p95_mm: float = 2.0,
    max_side_error_mm: float = 2.0,
    segment: bool = True,
) -> dict[str, Any]:
    """Pipeline completo sobre una foto: rectificación métrica por marcadores + segmentación.

    ``scale_correction`` multiplica el tamaño nominal del marcador para compensar una
    impresión que no salió al 100 % (por ejemplo, si la barra de control de 100 mm midió
    98,5 mm en el papel, pasar ``100/98.5``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if px_per_mm <= 0 or marker_size_mm <= 0 or scale_correction <= 0:
        raise ValueError("marker_size_mm, px_per_mm y scale_correction deben ser positivos")
    effective_marker_mm = marker_size_mm * scale_correction

    image = load_image(Path(image_path))
    detections = detect_aruco_markers(image, dictionary)
    corners_by_id = {d.marker_id: d.corners_image_px.astype(np.float64) for d in detections}

    rectification = metric_rectification_from_corners(corners_by_id, effective_marker_mm)
    homography = rectification["homography_image_to_mm"]
    gate = _self_consistency_gate(
        rectification["self_consistency"],
        max_corner_rms_mm,
        max_corner_p95_mm,
        max_side_error_mm,
    )

    span = rectification["self_consistency"]["marker_hull_span_mm"]
    margin_mm = marker_size_mm
    canvas_w = max(1, int(round((span[0] + 2 * margin_mm) * px_per_mm)))
    canvas_h = max(1, int(round((span[1] + 2 * margin_mm) * px_per_mm)))
    to_canvas = np.array(
        [[px_per_mm, 0.0, margin_mm * px_per_mm], [0.0, px_per_mm, margin_mm * px_per_mm], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    image_to_rectified_px = to_canvas @ homography
    rectified = cv2.warpPerspective(image, image_to_rectified_px, (canvas_w, canvas_h))

    per_marker_canvas = [
        {
            "marker_id": m["marker_id"],
            "corners_mm": (np.asarray(m["corners_mm"], np.float64) + margin_mm).tolist(),
        }
        for m in rectification["per_marker"]
    ]

    shapes: list[dict[str, Any]] = []
    segmentation_meta: dict[str, Any] = {}
    if segment:
        interior = _interior_from_markers(
            per_marker_canvas, marker_size_mm, px_per_mm, rectified.shape
        )
        mask, shapes, segmentation_meta = _segment_moulds(rectified, interior, px_per_mm)
        # contour_mm: mismo marco que homography_image_to_mm (origen = bbox de los marcadores).
        for shape in shapes:
            shape["contour_mm"] = [
                [round(px / px_per_mm - margin_mm, 3), round(py / px_per_mm - margin_mm, 3)]
                for px, py in shape["contour_px"]
            ]
        _write_image(output_dir / "shapes_mask.png", mask)
        _write_image(output_dir / "shapes_overlay.jpg", _shapes_overlay(rectified, shapes))

    _write_image(output_dir / "markers_overlay.jpg", _markers_overlay(image, detections, homography))
    _write_image(output_dir / "rectified_metric.jpg", rectified)

    report = {
        "schema_version": 1,
        "status": gate["decision"],
        "method": "metric_plane_rectification_from_equal_markers",
        "source_image": str(Path(image_path).resolve()),
        "dictionary": dictionary,
        "marker_size_mm": {
            "nominal": marker_size_mm,
            "scale_correction": scale_correction,
            "effective": round(effective_marker_mm, 5),
        },
        "px_per_mm": px_per_mm,
        "detected_marker_count": len(detections),
        "used_marker_ids": rectification["marker_ids"],
        # `image_to_board_mm` y `units` replican el esquema de aruco_stage1 para poder
        # pasar este JSON directo a scripts/validar_ground_truth.py.
        "units": {
            "source": "image_px",
            "processing": "image_px",
            "board": "board_mm",
            "rectified": "rectified_px",
        },
        "image_to_board_mm": np.round(homography, 12).tolist(),
        "homography_image_to_mm": np.round(homography, 12).tolist(),
        "image_to_rectified_px": np.round(image_to_rectified_px, 12).tolist(),
        "self_consistency": rectification["self_consistency"],
        "self_consistency_gate": gate,
        "per_marker": rectification["per_marker"],
        "moulds_mm": shapes,
        "segmentation": segmentation_meta,
        "limitations": [
            "El origen y la orientación del plano son arbitrarios (bbox de los marcadores).",
            "La escala depende de que los marcadores estén impresos al tamaño indicado; "
            "usar scale_correction si la barra de control no mide lo nominal.",
            "Sin corrección de lente: válido sólo si la distorsión es baja frente al objetivo.",
            "Este resultado pasó los chequeos internos de consistencia, pero todavía no fue "
            "comparado contra una pieza física ya medida a mano: no tomar estos números como "
            "definitivos hasta hacer esa comparación.",
        ],
    }
    (output_dir / "metric_rectification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "image": Path(image_path).name,
        "decision": gate["decision"],
        "detected_marker_count": len(detections),
        "used_marker_count": len(rectification["marker_ids"]),
        "corner_rms_mm": rectification["self_consistency"]["corner_rms_mm"],
        "corner_p95_mm": rectification["self_consistency"]["corner_p95_mm"],
        "recovered_side_mm": rectification["self_consistency"]["recovered_side_mm"],
        "mould_count": len(shapes),
        "moulds_mm": [
            {"id": s["id"], "width_mm": s["width_mm"], "height_mm": s["height_mm"]}
            for s in shapes
        ],
        "output_dir": str(output_dir),
    }
