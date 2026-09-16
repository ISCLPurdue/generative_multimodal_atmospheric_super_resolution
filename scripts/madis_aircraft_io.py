"""I/O helpers for MADIS point-aircraft observation files."""

from __future__ import annotations

import gzip
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def decode_to_temp(path: Path) -> Path:
    """Decompress one MADIS ``.gz`` file to a temporary NetCDF file."""
    tmp = tempfile.NamedTemporaryFile(
        prefix="madis_aircraft_", suffix=".nc", delete=False
    )
    tmp_path = Path(tmp.name)
    with gzip.open(path, "rb") as src:
        while chunk := src.read(1024 * 1024):
            tmp.write(chunk)
    tmp.close()
    return tmp_path


def read_var(ds: Any, *names: str) -> np.ndarray | None:
    """Read the first available variable and replace documented missing values."""
    for name in names:
        if name not in ds.variables:
            continue
        var = ds.variables[name]
        arr = np.asarray(var[:], dtype=np.float64)
        fill = getattr(var, "_FillValue", None)
        if fill is not None:
            arr = np.where(arr == float(fill), np.nan, arr)
        missing = getattr(var, "missing_value", None)
        if missing is not None:
            try:
                arr = np.where(arr == float(missing), np.nan, arr)
            except (TypeError, ValueError):
                pass
        arr = np.where(arr <= -9990.0, np.nan, arr)
        arr = np.where(arr >= 99990.0, np.nan, arr)
        return arr
    return None


def read_qcr(ds: Any, name: str, target_shape: tuple[int, ...]) -> np.ndarray:
    """Read a required MADIS quality-control result field."""
    qname = f"{name}QCR"
    if qname not in ds.variables:
        raise KeyError(
            f"Required MADIS quality-control field {qname!r} is absent"
        )
    arr = np.asarray(ds.variables[qname][:])
    if arr.shape == target_shape:
        return arr.astype(np.int64)
    if len(target_shape) == 2 and arr.ndim == 1 and arr.shape[0] == target_shape[0]:
        return np.repeat(arr[:, None], target_shape[1], axis=1).astype(np.int64)
    return np.broadcast_to(arr, target_shape).astype(np.int64)


def pressure_hpa_from_altitude_m(altitude_m: np.ndarray) -> np.ndarray:
    """Convert pressure altitude to hPa using the standard-atmosphere relation."""
    altitude = np.asarray(altitude_m, dtype=np.float64)
    pressure = 1013.25 * np.power(
        np.maximum(0.0, 1.0 - 2.25577e-5 * altitude), 5.25588
    )
    return np.where(np.isfinite(altitude), pressure, np.nan)


def to_uv(
    speed: np.ndarray | None, direction: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert meteorological wind speed and direction to zonal/meridional wind."""
    if speed is None or direction is None:
        return None, None
    radians = np.deg2rad(direction)
    return -speed * np.sin(radians), -speed * np.cos(radians)


def _broadcast_locations(
    latitude: np.ndarray,
    longitude: np.ndarray,
    target_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if latitude.shape == target_shape:
        return latitude, longitude
    if (
        len(target_shape) == 2
        and latitude.ndim == 1
        and latitude.shape[0] == target_shape[0]
    ):
        return (
            np.repeat(latitude[:, None], target_shape[1], axis=1),
            np.repeat(longitude[:, None], target_shape[1], axis=1),
        )
    return (
        np.broadcast_to(latitude, target_shape),
        np.broadcast_to(longitude, target_shape),
    )


def read_locations(
    ds: Any, target_shape: tuple[int, ...]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read latitude and longitude from the MADIS point-aircraft product."""
    latitude = read_var(ds, "latitude")
    longitude = read_var(ds, "longitude")
    if latitude is None or longitude is None:
        return None, None
    return _broadcast_locations(latitude, longitude, target_shape)


def parse_datetime(path: Path) -> datetime:
    """Parse the observation time from a MADIS archive filename."""
    match = re.search(r"(\d{8})_(\d{4})\.gz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse datetime from {path}")
    return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")
