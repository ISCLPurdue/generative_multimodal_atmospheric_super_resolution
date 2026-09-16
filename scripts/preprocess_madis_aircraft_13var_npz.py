#!/usr/bin/env python3
"""Build auditable per-timestep MADIS aircraft 13-var observation NPZ files."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import madis_aircraft_io
from observation_preprocessing_common import six_hour_index

DEFAULT_WINDOWS = {
    500: (475.0, 525.0),
    850: (825.0, 875.0),
}

VARIABLES = [
    ("temperature_500", "temperature", 500),
    ("u_component_of_wind_500", "u", 500),
    ("v_component_of_wind_500", "v", 500),
    ("temperature_850", "temperature", 850),
    ("u_component_of_wind_850", "u", 850),
    ("v_component_of_wind_850", "v", 850),
]


def lon_to_180(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


class VarStore:
    def __init__(self) -> None:
        self.lat: list[np.ndarray] = []
        self.lon: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []
        self.pressure: list[np.ndarray] = []
        self.source_product: list[np.ndarray] = []
        self.source_file_id: list[np.ndarray] = []
        self.data_source: list[np.ndarray] = []

    def append(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        vals: np.ndarray,
        pressure: np.ndarray,
        product: str,
        file_id: int,
        data_source: np.ndarray,
    ) -> None:
        n = int(vals.size)
        if n == 0:
            return
        self.lat.append(lat.astype(np.float32))
        self.lon.append(lon.astype(np.float32))
        self.vals.append(vals.astype(np.float32))
        self.pressure.append(pressure.astype(np.float32))
        self.source_product.append(np.asarray([product] * n, dtype="U16"))
        self.source_file_id.append(np.full(n, file_id, dtype=np.int16))
        self.data_source.append(data_source.astype(np.int16))

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.vals:
            return {
                "locs": np.empty((0, 2), dtype=np.float32),
                "vals": np.empty((0,), dtype=np.float32),
                "pressure_hpa_alt_derived": np.empty((0,), dtype=np.float32),
                "source_product": np.empty((0,), dtype="U16"),
                "source_file_id": np.empty((0,), dtype=np.int16),
                "data_source": np.empty((0,), dtype=np.int16),
                "qcr_pass": np.empty((0,), dtype=bool),
            }
        lat = np.concatenate(self.lat)
        lon = np.concatenate(self.lon)
        vals = np.concatenate(self.vals)
        pressure = np.concatenate(self.pressure)
        products = np.concatenate(self.source_product)
        file_ids = np.concatenate(self.source_file_id)
        data_sources = np.concatenate(self.data_source)
        return {
            "locs": np.stack([lat, lon], axis=1),
            "vals": vals,
            "pressure_hpa_alt_derived": pressure,
            "source_product": products,
            "source_file_id": file_ids,
            "data_source": data_sources,
            "qcr_pass": np.ones(vals.shape, dtype=bool),
        }


def parse_window(text: str) -> tuple[float, float]:
    parts = text.replace(":", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("window must be formatted like LO,HI")
    lo, hi = float(parts[0]), float(parts[1])
    if not lo < hi:
        raise argparse.ArgumentTypeError("window lower bound must be less than upper bound")
    return lo, hi


def process_file(
    path: Path,
    grouped: dict[int, dict[str, VarStore]],
    source_files: dict[int, list[str]],
    windows: dict[int, tuple[float, float]],
    exclude_data_sources: set[int],
    keep_data_sources: set[int],
) -> None:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise SystemExit("netCDF4 is required") from exc

    if "acarsProfiles" in str(path):
        raise ValueError(
            "This paper release uses the MADIS point/acars product only; "
            f"received {path}"
        )
    product = "acars"
    dt = madis_aircraft_io.parse_datetime(path)
    t_index = six_hour_index(dt)
    file_list = source_files[t_index]
    file_id = len(file_list)
    file_list.append(str(path))

    tmp = madis_aircraft_io.decode_to_temp(path)
    try:
        with Dataset(tmp, "r") as ds:
            temp = madis_aircraft_io.read_var(ds, "temperature")
            wind_speed = madis_aircraft_io.read_var(ds, "windSpeed")
            wind_dir = madis_aircraft_io.read_var(ds, "windDir")
            altitude = madis_aircraft_io.read_var(ds, "altitude")
            if temp is None or altitude is None:
                return
            data_source = madis_aircraft_io.read_var(ds, "dataSource")
            if data_source is None:
                data_source = np.full(temp.shape, -1, dtype=np.int16)
            else:
                data_source = np.asarray(data_source, dtype=np.int16)
            lat2, lon2 = madis_aircraft_io.read_locations(ds, temp.shape)
            if lat2 is None or lon2 is None:
                return
            lon2 = lon_to_180(lon2)
            pressure = madis_aircraft_io.pressure_hpa_from_altitude_m(altitude)
            u, v = madis_aircraft_io.to_uv(wind_speed, wind_dir)

            loc_qc = (
                (madis_aircraft_io.read_qcr(ds, "latitude", temp.shape) == 0)
                & (madis_aircraft_io.read_qcr(ds, "longitude", temp.shape) == 0)
                & (madis_aircraft_io.read_qcr(ds, "altitude", temp.shape) == 0)
            )
            base = (
                loc_qc
                & np.isfinite(lat2)
                & np.isfinite(lon2)
                & np.isfinite(pressure)
                & (pressure > 100.0)
                & (pressure < 1050.0)
            )
            if keep_data_sources:
                base = base & np.isin(data_source, list(keep_data_sources))
            if exclude_data_sources:
                base = base & (~np.isin(data_source, list(exclude_data_sources)))
            temp_qc = (madis_aircraft_io.read_qcr(ds, "temperature", temp.shape) == 0) & (temp > 180.0) & (temp < 330.0)
            wind_qc = (
                (madis_aircraft_io.read_qcr(ds, "windSpeed", temp.shape) == 0)
                & (madis_aircraft_io.read_qcr(ds, "windDir", temp.shape) == 0)
                & np.isfinite(wind_speed)
                & np.isfinite(wind_dir)
                & (wind_speed >= 0.0)
                & (wind_speed < 150.0)
            )
            obs_by_kind = {
                "temperature": temp,
                "u": u if u is not None else np.full_like(temp, np.nan),
                "v": v if v is not None else np.full_like(temp, np.nan),
            }
            qc_by_kind = {
                "temperature": temp_qc,
                "u": wind_qc,
                "v": wind_qc,
            }
            for variable, kind, level in VARIABLES:
                lo, hi = windows[level]
                obs = obs_by_kind[kind]
                mask = base & qc_by_kind[kind] & np.isfinite(obs) & (pressure >= lo) & (pressure <= hi)
                if not np.any(mask):
                    continue
                grouped[t_index][variable].append(
                    lat2[mask],
                    lon2[mask],
                    obs[mask],
                    pressure[mask],
                    product,
                    file_id,
                    data_source[mask],
                )
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def write_outputs(
    grouped: dict[int, dict[str, VarStore]],
    source_files: dict[int, list[str]],
    out_root: Path,
    windows: dict[int, tuple[float, float]],
    exclude_data_sources: set[int],
    keep_data_sources: set[int],
) -> None:
    obs_dir = out_root / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t_index in sorted(grouped):
        payload: dict[str, Any] = {}
        for variable in [v[0] for v in VARIABLES]:
            arrays = grouped[t_index][variable].arrays()
            for suffix, arr in arrays.items():
                payload[f"{variable}_{suffix}"] = arr
            rows.append(
                {
                    "timestep": f"t{t_index:04d}",
                    "era5_timestep_index": t_index,
                    "variable": variable,
                    "n_obs": int(arrays["vals"].size),
                    "n_acars": int(np.sum(arrays["source_product"] == "acars")) if arrays["vals"].size else 0,
                }
            )
        metadata = {
            "era5_timestep_index": t_index,
            "pressure_windows_hpa": windows,
            "location_protocol": "MADIS point/acars latitude and longitude",
            "pressure_protocol": "pressure_hpa_alt_derived from MADIS pressure altitude using standard-atmosphere conversion",
            "qcr_policy": (
                "kept observations require QCR==0 for the MADIS latitude, "
                "longitude, altitude, and applicable temperature or wind fields; "
                "finite-coordinate and physical-range filters are also applied"
            ),
            "excluded_data_sources": sorted(exclude_data_sources),
            "kept_data_sources": sorted(keep_data_sources),
        }
        payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype="U")
        payload["source_files"] = np.asarray(source_files[t_index], dtype="U512")
        np.savez_compressed(obs_dir / f"madis_aircraft_13var_t{t_index:04d}.npz", **payload)

    summary_path = out_root / "madis_aircraft_13var_npz_summary.csv"
    import csv

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestep",
                "era5_timestep_index",
                "variable",
                "n_obs",
                "n_acars",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "README.md").write_text(
        "# MADIS Aircraft 13-var NPZ Observations\n\n"
        "This directory contains auditable per-timestep aircraft observation NPZ files.\n\n"
        "- Source product: MADIS `point/acars`\n"
        f"- Pressure windows: 500 hPa = {windows[500][0]:g}-{windows[500][1]:g}, "
        f"850 hPa = {windows[850][0]:g}-{windows[850][1]:g}\n"
        "- Pressure field: `pressure_hpa_alt_derived`, derived from MADIS pressure altitude\n"
        "- Location fields: MADIS `latitude` and `longitude`\n"
        f"- Kept MADIS dataSource codes: {sorted(keep_data_sources) if keep_data_sources else 'all except excluded'}\n"
        f"- Excluded MADIS dataSource codes: {sorted(exclude_data_sources)}\n"
        "- Timestep indices are zero-based six-hour indices within each source file's calendar year.\n"
        f"- Summary CSV: `{summary_path}`\n"
    )
    print(f"Wrote {obs_dir}")
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--window-500", type=parse_window, default=DEFAULT_WINDOWS[500], help="Pressure window for 500 hPa; paper default: 475,525")
    parser.add_argument("--window-850", type=parse_window, default=DEFAULT_WINDOWS[850], help="Pressure window for 850 hPa; paper default: 825,875")
    parser.add_argument(
        "--exclude-data-sources",
        default="",
        help="Comma-separated MADIS dataSource integer codes to exclude, e.g. 4,7,8 for TAMDAR streams.",
    )
    parser.add_argument(
        "--keep-data-sources",
        default="",
        help="Comma-separated MADIS dataSource integer codes to keep. If set, all other sources are dropped before excludes.",
    )
    args = parser.parse_args()

    windows = {500: args.window_500, 850: args.window_850}
    exclude_data_sources = {
        int(x.strip()) for x in args.exclude_data_sources.split(",") if x.strip()
    }
    keep_data_sources = {
        int(x.strip()) for x in args.keep_data_sources.split(",") if x.strip()
    }
    grouped: dict[int, dict[str, VarStore]] = defaultdict(lambda: defaultdict(VarStore))
    source_files: dict[int, list[str]] = defaultdict(list)
    for path in args.files:
        process_file(path, grouped, source_files, windows, exclude_data_sources, keep_data_sources)
    write_outputs(grouped, source_files, args.out_root, windows, exclude_data_sources, keep_data_sources)


if __name__ == "__main__":
    main()
