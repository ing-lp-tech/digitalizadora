from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraProfile:
    name: str
    image_size_px: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    camera_model: str = "opencv_pinhole"
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image_size_px = tuple(map(int, self.image_size_px))
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)
        if self.schema_version != 1:
            raise ValueError(f"Versión de perfil no soportada: {self.schema_version}")
        if self.camera_model != "opencv_pinhole":
            raise ValueError(f"Modelo de cámara no soportado: {self.camera_model}")
        if min(self.image_size_px) <= 0:
            raise ValueError("La resolución del perfil debe ser positiva")
        if not np.isfinite(self.camera_matrix).all() or not np.isfinite(self.dist_coeffs).all():
            raise ValueError("El perfil contiene valores no finitos")
        if self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0:
            raise ValueError("Las distancias focales deben ser positivas")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "camera_model": self.camera_model,
            "image_size_px": list(self.image_size_px),
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.ravel().tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraProfile":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            name=str(data["name"]),
            camera_model=str(data.get("camera_model", "opencv_pinhole")),
            image_size_px=tuple(data["image_size_px"]),
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: Path) -> "CameraProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def intrinsics_for(self, image_size_px: tuple[int, int]) -> np.ndarray:
        target_width, target_height = map(int, image_size_px)
        source_width, source_height = self.image_size_px
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        if abs(scale_x - scale_y) > max(scale_x, scale_y) * 0.002:
            raise ValueError(
                "La imagen no conserva la relación de aspecto del perfil de cámara; "
                "se necesita un perfil específico para esa resolución/modo"
            )
        scaled = self.camera_matrix.copy()
        scaled[0, :] *= scale_x
        scaled[1, :] *= scale_y
        scaled[2, :] = self.camera_matrix[2, :]
        return scaled


def undistort_image(
    image: np.ndarray,
    profile: CameraProfile,
    alpha: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    camera_matrix = profile.intrinsics_for((width, height))
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        profile.dist_coeffs,
        (width, height),
        float(alpha),
        (width, height),
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        profile.dist_coeffs,
        None,
        new_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    corrected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return corrected, {
        "profile_name": profile.name,
        "profile_image_size_px": list(profile.image_size_px),
        "input_image_size_px": [width, height],
        "alpha": float(alpha),
        "scaled_camera_matrix": np.round(camera_matrix, 12).tolist(),
        "new_camera_matrix": np.round(new_matrix, 12).tolist(),
        "valid_roi_px": list(map(int, roi)),
    }


def undistort_points_to_pixels(
    points: np.ndarray,
    profile: CameraProfile,
    image_size_px: tuple[int, int] | None = None,
    output_camera_matrix: np.ndarray | None = None,
) -> np.ndarray:
    size = image_size_px or profile.image_size_px
    camera_matrix = profile.intrinsics_for(size)
    output = camera_matrix if output_camera_matrix is None else np.asarray(output_camera_matrix, np.float64)
    return cv2.undistortPoints(
        np.asarray(points, np.float32).reshape(-1, 1, 2),
        camera_matrix,
        profile.dist_coeffs,
        P=output,
    ).reshape(-1, 2)

