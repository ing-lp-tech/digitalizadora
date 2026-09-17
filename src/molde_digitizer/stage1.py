from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .camera import CameraProfile, undistort_image


@dataclass
class Candidate:
    method: str
    quad: np.ndarray
    rectified: np.ndarray | None = None
    homography: np.ndarray | None = None
    references: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None
    score: float = -1.0


def load_image(path: Path) -> np.ndarray:
    """Carga respetando EXIF y devuelve BGR, sin depender de rutas ASCII."""
    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)
    # Orden esperado: superior izquierda, superior derecha, inferior derecha, inferior izquierda.
    first_edge = pts[1] - pts[0]
    second_edge = pts[2] - pts[1]
    signed_cross = first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
    if signed_cross < 0:
        pts = pts[[0, 3, 2, 1]]
    return pts.astype(np.float32)


def _find_board_contour(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    short = min(gray.shape)
    blur_size = max(5, int(round(short * 0.007)) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    threshold, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    close_size = max(5, int(round(short * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    viable = [
        contour
        for contour in contours
        if 0.20 <= cv2.contourArea(contour) / image_area <= 0.98
    ]
    if not viable:
        raise RuntimeError("No se encontró una región oscura compatible con la pizarra")

    def contour_score(contour: np.ndarray) -> float:
        area_ratio = cv2.contourArea(contour) / image_area
        x, y, w, h = cv2.boundingRect(contour)
        center = np.array([x + w / 2, y + h / 2])
        image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])
        offset = np.linalg.norm((center - image_center) / np.array(image.shape[1::-1]))
        return area_ratio - 0.15 * offset

    contour = max(viable, key=contour_score)
    return contour, mask, float(threshold)


def _polygon_quad(contour: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in np.linspace(0.006, 0.08, 60):
        approx = cv2.approxPolyDP(hull, float(epsilon) * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad(approx[:, 0, :])
    raise RuntimeError("El contorno de la pizarra no pudo reducirse a cuatro esquinas")


def _min_area_quad(contour: np.ndarray) -> np.ndarray:
    return order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))


def _warp(image: np.ndarray, quad: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    tl, tr, br, bl = order_quad(quad)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    scale = min(1.0, max_side / max(width, height))
    out_w = max(320, int(round(width * scale)))
    out_h = max(240, int(round(height * scale)))
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(order_quad(quad), destination)
    rectified = cv2.warpPerspective(image, homography, (out_w, out_h))
    return rectified, homography


def _deduplicate_circles(circles: list[tuple[float, float, float]], distance: float) -> list[tuple[float, float, float]]:
    accepted: list[tuple[float, float, float]] = []
    for circle in sorted(circles, key=lambda item: item[2], reverse=True):
        if all(math.hypot(circle[0] - other[0], circle[1] - other[1]) >= distance for other in accepted):
            accepted.append(circle)
    return accepted


def _circle_photometry(gray: np.ndarray, x: float, y: float, radius: float) -> tuple[float, float]:
    height, width = gray.shape
    extent = int(math.ceil(radius * 1.9))
    x0, x1 = max(0, int(x) - extent), min(width, int(x) + extent + 1)
    y0, y1 = max(0, int(y) - extent), min(height, int(y) + extent + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return -255.0, 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt(np.square(xx - x) + np.square(yy - y))
    inner_mask = distance <= radius * 1.05
    outer_mask = (distance >= radius * 1.25) & (distance <= radius * 1.8)
    inner = gray[y0:y1, x0:x1][inner_mask]
    outer = gray[y0:y1, x0:x1][outer_mask]
    if inner.size == 0 or outer.size == 0:
        return -255.0, 0.0
    contrast = float(np.mean(inner) - np.median(outer))
    bright_fraction = float(np.mean(inner >= max(145.0, np.median(outer) + 35.0)))
    return contrast, bright_fraction


def _detect_references(rectified: np.ndarray) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    short = min(width, height)
    blurred = cv2.medianBlur(gray, 5)
    raw: list[tuple[float, float, float]] = []
    min_radius = max(4, int(short * 0.0035))
    max_radius = max(min_radius + 2, int(short * 0.025))
    min_distance = max(10, int(short * 0.012))
    # Dos sensibilidades: la estricta encuentra aros completos; la flexible recupera medios círculos.
    for param2 in (24, 18, 14):
        found = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_distance,
            param1=120,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if found is not None:
            raw.extend(tuple(map(float, circle)) for circle in found[0])

    circles = _deduplicate_circles(raw, max(12.0, short * 0.014))
    references: list[dict[str, Any]] = []
    band = 0.105
    for x, y, radius in circles:
        normalized = np.array([x / width, y / height])
        distances = {
            "top": normalized[1],
            "right": 1.0 - normalized[0],
            "bottom": 1.0 - normalized[1],
            "left": normalized[0],
        }
        side = min(distances, key=distances.get)
        if distances[side] > band:
            continue
        # Se descartan esquinas externas y detecciones pegadas al recorte.
        if not (0.006 < normalized[0] < 0.994 and 0.006 < normalized[1] < 0.994):
            continue
        feature_contrast, bright_fraction = _circle_photometry(gray, x, y, radius)
        # Referencia blanca sobre fondo oscuro. Esta prueba elimina imanes oscuros
        # rodeados por papel claro y tornillos del marco metálico.
        if feature_contrast < 12.0 or bright_fraction < 0.12:
            continue
        references.append(
            {
                "rectified_px": [round(x, 3), round(y, 3)],
                "radius_px": round(radius, 3),
                "side": side,
                "distance_to_border_norm": round(float(distances[side]), 6),
                "feature_contrast": round(feature_contrast, 3),
                "bright_fraction": round(bright_fraction, 4),
            }
        )

    ordering = {"top": 0, "right": 1, "bottom": 2, "left": 3}
    references.sort(
        key=lambda ref: (
            ordering[ref["side"]],
            ref["rectified_px"][0] if ref["side"] in ("top", "bottom") else ref["rectified_px"][1],
        )
    )
    return references


def _reference_metrics(references: list[dict[str, Any]], shape: tuple[int, ...]) -> dict[str, Any]:
    height, width = shape[:2]
    by_side: dict[str, list[dict[str, Any]]] = {side: [] for side in ("top", "right", "bottom", "left")}
    for reference in references:
        by_side[reference["side"]].append(reference)

    residuals: list[float] = []
    spacing_cvs: list[float] = []
    side_details: dict[str, Any] = {}
    for side, items in by_side.items():
        across_index = 0 if side in ("top", "bottom") else 1
        normal_index = 1 - across_index
        normal_size = height if normal_index == 1 else width
        coordinates = np.array([item["rectified_px"] for item in items], dtype=np.float64)
        detail: dict[str, Any] = {"count": len(items)}
        if len(items) >= 3:
            normal = coordinates[:, normal_index]
            median = float(np.median(normal))
            rms = float(np.sqrt(np.mean(np.square(normal - median))))
            residuals.extend(((normal - median) / normal_size).tolist())
            across = np.sort(coordinates[:, across_index])
            gaps = np.diff(across)
            if len(gaps) >= 2 and np.mean(gaps) > 0:
                spacing_cv = float(np.std(gaps) / np.mean(gaps))
                spacing_cvs.append(spacing_cv)
                detail["spacing_cv"] = round(spacing_cv, 5)
            detail["line_rms_px"] = round(rms, 4)
            detail["line_position_px"] = round(median, 3)
        side_details[side] = detail

    normalized_rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 1.0
    return {
        "reference_count": len(references),
        "side_counts": {side: len(items) for side, items in by_side.items()},
        "all_sides_have_minimum": all(len(items) >= 3 for items in by_side.values()),
        "reference_line_rms_normalized": round(normalized_rms, 7),
        "spacing_cv_mean": round(float(np.mean(spacing_cvs)), 5) if spacing_cvs else None,
        "sides": side_details,
    }


def _edge_contrast(image: np.ndarray, quad: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center = order_quad(quad).mean(axis=0)
    inside: list[float] = []
    outside: list[float] = []
    ordered = order_quad(quad)
    for start, end in zip(ordered, np.roll(ordered, -1, axis=0)):
        for fraction in np.linspace(0.1, 0.9, 30):
            point = start * (1 - fraction) + end * fraction
            direction = center - point
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            unit = direction / norm
            for target, sign in ((inside, 1), (outside, -1)):
                sample = point + sign * unit * max(4, min(gray.shape) * 0.008)
                x = int(np.clip(round(sample[0]), 0, gray.shape[1] - 1))
                y = int(np.clip(round(sample[1]), 0, gray.shape[0] - 1))
                target.append(float(gray[y, x]))
    if not inside or not outside:
        return 0.0
    # La pizarra debería ser más oscura que el exterior.
    return float(np.clip((np.median(outside) - np.median(inside)) / 255.0, -1.0, 1.0))


def _candidate_score(metrics: dict[str, Any]) -> float:
    count_score = min(1.0, metrics["reference_count"] / 40.0)
    side_score = sum(min(1.0, count / 5.0) for count in metrics["side_counts"].values()) / 4.0
    rms = metrics["reference_line_rms_normalized"]
    alignment_score = max(0.0, 1.0 - rms / 0.015)
    contrast_score = max(0.0, metrics["edge_contrast"])
    spacing_cv = metrics.get("spacing_cv_mean")
    spacing_score = max(0.0, 1.0 - spacing_cv / 0.65) if spacing_cv is not None else 0.0
    return round(
        0.25 * count_score
        + 0.20 * side_score
        + 0.25 * alignment_score
        + 0.15 * spacing_score
        + 0.15 * contrast_score,
        6,
    )


def _capture_metrics(image: np.ndarray, quad: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    ordered = order_quad(quad)
    margins = np.column_stack(
        (
            ordered[:, 0] / width,
            (width - 1 - ordered[:, 0]) / width,
            ordered[:, 1] / height,
            (height - 1 - ordered[:, 1]) / height,
        )
    )
    return {
        "blur_laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 5),
        "near_black_fraction": round(float(np.mean(gray <= 5)), 7),
        "near_white_fraction": round(float(np.mean(gray >= 250)), 7),
        "quad_min_image_margin_normalized": round(float(np.min(margins)), 7),
    }


def _capture_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    rejection_reasons: list[str] = []
    warnings: list[str] = []
    if metrics["reference_count"] < 12:
        rejection_reasons.append("Se detectaron menos de 12 referencias periféricas")
    weak_sides = [side for side, count in metrics["side_counts"].items() if count < 3]
    if weak_sides:
        rejection_reasons.append("Lados sin referencias suficientes: " + ", ".join(weak_sides))
    if metrics["reference_line_rms_normalized"] > 0.015:
        rejection_reasons.append("Las referencias no quedan alineadas después de rectificar")
    if metrics["edge_contrast"] < 0.05:
        rejection_reasons.append("El borde de la pizarra no tiene contraste suficiente")
    if metrics["quad_min_image_margin_normalized"] < 0.005:
        rejection_reasons.append("La pizarra toca el borde de la imagen o está cortada")
    if metrics["blur_laplacian_variance"] < 25.0:
        rejection_reasons.append("La fotografía está demasiado desenfocada")
    elif metrics["blur_laplacian_variance"] < 55.0:
        warnings.append("Nitidez baja; conviene repetir la captura")
    if metrics["near_white_fraction"] > 0.12:
        warnings.append("Hay una fracción alta de píxeles saturados")
    return {
        "decision": "reject" if rejection_reasons else ("warn" if warnings else "pass"),
        "rejection_reasons": rejection_reasons,
        "warnings": warnings,
        "scope": "Valida aptitud para rectificación en píxeles; no valida milímetros",
    }


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 3:
        return None
    kept = np.asarray(points, dtype=np.float32)
    for _ in range(3):
        vx, vy, x0, y0 = cv2.fitLine(kept, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        direction = np.array([vx, vy], dtype=np.float64)
        origin = np.array([x0, y0], dtype=np.float64)
        distances = np.abs(np.cross(direction, kept - origin))
        median = np.median(distances)
        mad = np.median(np.abs(distances - median))
        limit = max(2.0, median + 3.5 * max(mad, 0.5))
        filtered = kept[distances <= limit]
        if len(filtered) == len(kept) or len(filtered) < 3:
            break
        kept = filtered
    return origin, direction / np.linalg.norm(direction)


def _line_intersection(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    p, r = first
    q, s = second
    cross = float(np.cross(r, s))
    if abs(cross) < 1e-8:
        raise RuntimeError("Líneas de referencias paralelas o ambiguas")
    t = float(np.cross(q - p, s) / cross)
    return (p + t * r).astype(np.float32)


def _refined_quad(candidate: Candidate) -> np.ndarray | None:
    if candidate.homography is None or not candidate.references or candidate.rectified is None:
        return None
    by_side: dict[str, list[np.ndarray]] = {side: [] for side in ("top", "right", "bottom", "left")}
    inverse = np.linalg.inv(candidate.homography)
    for reference in candidate.references:
        rectified_point = np.array([[reference["rectified_px"]]], dtype=np.float32)
        source_point = cv2.perspectiveTransform(rectified_point, inverse)[0, 0]
        by_side[reference["side"]].append(source_point)
    lines = {side: _fit_line(np.array(points)) for side, points in by_side.items()}
    if any(line is None for line in lines.values()):
        return None
    source_ref = np.array(
        [
            _line_intersection(lines["top"], lines["left"]),
            _line_intersection(lines["top"], lines["right"]),
            _line_intersection(lines["bottom"], lines["right"]),
            _line_intersection(lines["bottom"], lines["left"]),
        ],
        dtype=np.float32,
    )
    metrics = candidate.metrics or {}
    sides = metrics.get("sides", {})
    try:
        left = float(sides["left"]["line_position_px"])
        right = float(sides["right"]["line_position_px"])
        top = float(sides["top"]["line_position_px"])
        bottom = float(sides["bottom"]["line_position_px"])
    except (KeyError, TypeError):
        return None
    destination_ref = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32
    )
    source_to_rectified = cv2.getPerspectiveTransform(source_ref, destination_ref)
    height, width = candidate.rectified.shape[:2]
    outer_rectified = np.array(
        [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
        dtype=np.float32,
    )
    refined = cv2.perspectiveTransform(outer_rectified, np.linalg.inv(source_to_rectified))[0]
    return order_quad(refined)


def _evaluate_candidate(image: np.ndarray, candidate: Candidate) -> Candidate:
    candidate.rectified, candidate.homography = _warp(image, candidate.quad)
    candidate.references = _detect_references(candidate.rectified)
    metrics = _reference_metrics(candidate.references, candidate.rectified.shape)
    metrics["edge_contrast"] = round(_edge_contrast(image, candidate.quad), 6)
    metrics["quad_area_ratio"] = round(
        abs(cv2.contourArea(order_quad(candidate.quad))) / (image.shape[0] * image.shape[1]), 6
    )
    metrics.update(_capture_metrics(image, candidate.quad))
    candidate.metrics = metrics
    candidate.score = _candidate_score(metrics)
    return candidate


def _overlay(image: np.ndarray, candidate: Candidate) -> np.ndarray:
    output = image.copy()
    quad = np.round(order_quad(candidate.quad)).astype(np.int32)
    cv2.polylines(output, [quad], True, (40, 220, 40), max(2, min(image.shape[:2]) // 500))
    if candidate.homography is not None and candidate.references:
        inverse = np.linalg.inv(candidate.homography)
        for index, reference in enumerate(candidate.references, start=1):
            rectified = np.array([[reference["rectified_px"]]], dtype=np.float32)
            point = cv2.perspectiveTransform(rectified, inverse)[0, 0]
            center = tuple(np.round(point).astype(int))
            radius = max(4, int(round(reference["radius_px"])))
            color = {"top": (255, 80, 80), "right": (80, 180, 255), "bottom": (80, 255, 100), "left": (255, 80, 255)}[reference["side"]]
            cv2.circle(output, center, radius, color, 2)
            if index <= 99:
                cv2.putText(output, str(index), (center[0] + 3, center[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    label = f"{candidate.method} | score={candidate.score:.3f} | refs={len(candidate.references or [])}"
    cv2.rectangle(output, (10, 10), (min(output.shape[1] - 10, 760), 54), (0, 0, 0), -1)
    cv2.putText(output, label, (20, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"No se pudo codificar {path.name}")
    encoded.tofile(path)


def _json_ready(candidate: Candidate) -> dict[str, Any]:
    return {
        "method": candidate.method,
        "score": candidate.score,
        "quad_image_px": np.round(order_quad(candidate.quad), 4).tolist(),
        "homography_image_to_rectified": np.round(candidate.homography, 10).tolist() if candidate.homography is not None else None,
        "rectified_size_px": [candidate.rectified.shape[1], candidate.rectified.shape[0]] if candidate.rectified is not None else None,
        "metrics": candidate.metrics,
        "references": candidate.references,
    }


def process_stage1(
    image_path: Path,
    output_dir: Path,
    camera_profile_path: Path | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_image = load_image(image_path)
    image = original_image
    camera_correction: dict[str, Any] = {
        "applied": False,
        "profile_path": None,
        "reason": "No se proporcionó un perfil de cámara",
    }
    if camera_profile_path is not None:
        profile = CameraProfile.load(camera_profile_path)
        image, correction_details = undistort_image(original_image, profile, alpha=0.0)
        camera_correction = {
            "applied": True,
            "profile_path": str(Path(camera_profile_path).resolve()),
            **correction_details,
        }
        _write_image(output_dir / "undistorted.jpg", image)
    contour, dark_mask, otsu_threshold = _find_board_contour(image)

    candidates = [
        _evaluate_candidate(image, Candidate("dark_contour_polygon", _polygon_quad(contour))),
        _evaluate_candidate(image, Candidate("dark_min_area_rectangle", _min_area_quad(contour))),
    ]
    refined = _refined_quad(candidates[0])
    if refined is not None:
        candidates.append(_evaluate_candidate(image, Candidate("fiducial_line_refinement", refined)))
    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]

    detections = {
        "schema_version": 1,
        "source_image": str(image_path.resolve()),
        "source_size_px": [original_image.shape[1], original_image.shape[0]],
        "processing_size_px": [image.shape[1], image.shape[0]],
        "units": {
            "source": "image_px",
            "processing": "undistorted_px" if camera_correction["applied"] else "image_px",
            "rectified": "rectified_px_arbitrary_scale",
            "physical": None,
        },
        "camera_correction": camera_correction,
        "selected_method": best.method,
        "candidates": [_json_ready(candidate) for candidate in candidates],
    }
    quality = {
        "schema_version": 1,
        "status": "experimental_geometry_only",
        "selected_method": best.method,
        "selected_score": best.score,
        "metrics": best.metrics,
        "capture_gate": _capture_gate(best.metrics or {}),
        "physical_scale_available": False,
        "lens_profile_available": bool(camera_correction["applied"]),
        "camera_correction": camera_correction,
        "millimeter_accuracy_validated": False,
        "limitations": [
            "La imagen no contiene EXIF ni identificación de cámara/lente."
            if not camera_correction["applied"]
            else "El perfil aplicado todavía debe validarse contra el celular físico.",
            "No se conocen las dimensiones físicas ni las coordenadas de las referencias.",
            "Las referencias actuales no tienen identidad única; sólo se ordenan por lado.",
            "La rectificación está en píxeles y no certifica todavía una escala en milímetros.",
        ],
    }

    (output_dir / "detections.json").write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_image(output_dir / "dark_mask.png", dark_mask)
    _write_image(output_dir / "references_overlay.jpg", _overlay(image, best))
    if best.rectified is None:
        raise RuntimeError("El candidato seleccionado no produjo una imagen rectificada")
    _write_image(output_dir / "rectified.jpg", best.rectified)
    for candidate in candidates:
        _write_image(output_dir / f"candidate_{candidate.method}.jpg", _overlay(image, candidate))

    return {
        "image": image_path.name,
        "output_dir": str(output_dir),
        "otsu_threshold": round(otsu_threshold, 3),
        "selected_method": best.method,
        "score": best.score,
        "reference_count": len(best.references or []),
        "side_counts": best.metrics["side_counts"] if best.metrics else {},
        "reference_line_rms_normalized": best.metrics["reference_line_rms_normalized"] if best.metrics else None,
        "capture_gate": _capture_gate(best.metrics or {}),
        "candidates": [
            {
                "method": candidate.method,
                "score": candidate.score,
                "reference_count": len(candidate.references or []),
                "reference_line_rms_normalized": candidate.metrics["reference_line_rms_normalized"] if candidate.metrics else None,
            }
            for candidate in candidates
        ],
    }


def reprojection_metrics(source: np.ndarray, destination: np.ndarray, homography: np.ndarray) -> dict[str, float]:
    projected = cv2.perspectiveTransform(np.asarray(source, np.float32).reshape(1, -1, 2), homography)[0]
    errors = np.linalg.norm(projected - np.asarray(destination, np.float32), axis=1)
    return {
        "rms": float(np.sqrt(np.mean(np.square(errors)))),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }
