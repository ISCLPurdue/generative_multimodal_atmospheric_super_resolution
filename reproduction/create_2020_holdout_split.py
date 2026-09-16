#!/usr/bin/env python3
"""Create the paper's retained-80% and excluded-20% observation split.

Surface observations are split by the union of observed ERA5 cells at each
analysis time.  Aircraft observations are split independently for each of the
six aircraft-targeted variables, so the sampling unit is ERA5 cell x variable.
The random generator is seeded with ``base_seed + timestep`` and the same
generator is used in the fixed variable order below.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMESTEPS = (
    REPO_ROOT / "reproduction/manifests/holdout_timesteps_2020.json"
)
DEFAULT_INTERFACE = REPO_ROOT / "reproduction/config/selected_interface_2019.json"
CONUS_BOUNDS = (24.0, 50.0, -125.0, -66.0)
AIRCRAFT_VARIABLES = [
    "temperature_500",
    "temperature_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
]
SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]


def nearest_era5_flat_cells(
    locations: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray
) -> np.ndarray:
    """Map latitude/longitude pairs to nearest ERA5 flat-cell indices."""
    dlat = float(np.median(np.diff(lat_axis)))
    dlon = float(np.median(np.diff(lon_axis)))
    locations = np.asarray(locations, dtype=np.float64)
    lat_index = np.rint(
        (locations[:, 0] - float(lat_axis[0])) / dlat
    ).astype(np.int64)
    lon_index = np.rint(
        (np.mod(locations[:, 1], 360.0) - float(lon_axis[0])) / dlon
    ).astype(np.int64)
    lat_index = np.clip(lat_index, 0, lat_axis.size - 1)
    lon_index = np.mod(lon_index, lon_axis.size)
    return lat_index * lon_axis.size + lon_index


def conus_mask(locations: np.ndarray) -> np.ndarray:
    """Select reports by native coordinates within the paper's CONUS domain."""
    locations = np.asarray(locations, dtype=np.float64)
    if locations.size == 0:
        return np.zeros((0,), dtype=bool)
    lat0, lat1, lon0, lon1 = CONUS_BOUNDS
    longitude = (locations[:, 1] + 180.0) % 360.0 - 180.0
    return (
        np.isfinite(locations[:, 0])
        & np.isfinite(longitude)
        & (locations[:, 0] >= lat0)
        & (locations[:, 0] <= lat1)
        & (longitude >= lon0)
        & (longitude <= lon1)
    )


def load_metadata(data: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in data.files:
        return {}
    raw = str(data["metadata_json"])
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_metadata_json": raw}


def choose_cells(
    cells: np.ndarray, rng: np.random.Generator, holdout_fraction: float
) -> np.ndarray:
    unique_cells = np.unique(cells)
    if unique_cells.size == 0:
        return np.empty((0,), dtype=np.int64)
    count = max(1, int(round(holdout_fraction * unique_cells.size)))
    count = min(count, unique_cells.size)
    return np.sort(rng.choice(unique_cells, size=count, replace=False))


def subset_variable_arrays(
    data: np.lib.npyio.NpzFile,
    variable: str,
    is_retained: np.ndarray,
    is_excluded: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Subset every observation-level array belonging to one variable."""
    retained: dict[str, np.ndarray] = {}
    excluded: dict[str, np.ndarray] = {}
    observation_count = is_retained.size
    prefix = f"{variable}_"
    for key in data.files:
        if key.startswith(prefix):
            array = np.asarray(data[key])
            if array.shape[:1] == (observation_count,):
                retained[key] = array[is_retained]
                excluded[key] = array[is_excluded]
            else:
                retained[key] = array
                excluded[key] = array
    return retained, excluded


def split_aircraft(
    data: np.lib.npyio.NpzFile,
    timestep: int,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    seed: int,
    holdout_fraction: float,
    allowed_source_codes: set[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed + timestep)
    retained: dict[str, np.ndarray] = {}
    excluded: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    variable_prefixes = tuple(f"{variable}_" for variable in AIRCRAFT_VARIABLES)
    for key in data.files:
        if not key.startswith(variable_prefixes):
            retained[key] = np.asarray(data[key])
            excluded[key] = np.asarray(data[key])

    for variable in AIRCRAFT_VARIABLES:
        all_locations = np.asarray(data[f"{variable}_locs"], dtype=np.float32)
        source_codes = np.asarray(data[f"{variable}_data_source"], dtype=np.int16)
        source_products = np.asarray(data[f"{variable}_source_product"])
        is_eligible = (
            np.isin(source_codes, sorted(allowed_source_codes))
            & (source_products == "acars")
            & conus_mask(all_locations)
        )
        locations = all_locations[is_eligible]
        cells = (
            nearest_era5_flat_cells(locations, lat_axis, lon_axis)
            if locations.size
            else np.empty((0,), dtype=np.int64)
        )
        excluded_cells = choose_cells(cells, rng, holdout_fraction)
        excluded_set = set(int(cell) for cell in excluded_cells)
        selected_is_excluded = np.asarray(
            [int(cell) in excluded_set for cell in cells], dtype=bool
        )
        is_retained = np.zeros(all_locations.shape[0], dtype=bool)
        is_excluded = np.zeros(all_locations.shape[0], dtype=bool)
        eligible_indices = np.flatnonzero(is_eligible)
        is_retained[eligible_indices[~selected_is_excluded]] = True
        is_excluded[eligible_indices[selected_is_excluded]] = True
        variable_retained, variable_excluded = subset_variable_arrays(
            data, variable, is_retained, is_excluded
        )
        retained.update(variable_retained)
        excluded.update(variable_excluded)
        excluded[f"{variable}_era5_flat_cells"] = cells[selected_is_excluded]
        rows.append(
            {
                "timestep": timestep,
                "source": "aircraft",
                "variable": variable,
                "spatial_domain": "CONUS by native report coordinates",
                "holdout_unit": "ERA5 cell x variable",
                "n_input_obs": int(all_locations.shape[0]),
                "n_original_obs": int(cells.size),
                "n_retained_obs": int(is_retained.sum()),
                "n_excluded_obs": int(is_excluded.sum()),
                "n_original_cells": int(np.unique(cells).size),
                "n_excluded_cells": int(excluded_cells.size),
            }
        )
    return retained, excluded, rows


def split_surface(
    data: np.lib.npyio.NpzFile,
    timestep: int,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    seed: int,
    holdout_fraction: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    per_variable_cells: dict[str, np.ndarray] = {}
    per_variable_eligible: dict[str, np.ndarray] = {}
    observed_cells: list[np.ndarray] = []
    for variable in SURFACE_VARIABLES:
        locations = np.asarray(data[f"{variable}_locs"], dtype=np.float32)
        is_eligible = conus_mask(locations)
        eligible_locations = locations[is_eligible]
        cells = (
            nearest_era5_flat_cells(eligible_locations, lat_axis, lon_axis)
            if eligible_locations.size
            else np.empty((0,), dtype=np.int64)
        )
        per_variable_cells[variable] = cells
        per_variable_eligible[variable] = is_eligible
        if cells.size:
            observed_cells.append(np.unique(cells))

    union_cells = (
        np.unique(np.concatenate(observed_cells))
        if observed_cells
        else np.empty((0,), dtype=np.int64)
    )
    rng = np.random.default_rng(seed + timestep)
    excluded_cells = choose_cells(union_cells, rng, holdout_fraction)
    excluded_set = set(int(cell) for cell in excluded_cells)
    retained: dict[str, np.ndarray] = {}
    excluded: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for variable in SURFACE_VARIABLES:
        cells = per_variable_cells[variable]
        eligible_is_excluded = np.asarray(
            [int(cell) in excluded_set for cell in cells], dtype=bool
        )
        locations = np.asarray(data[f"{variable}_locs"], dtype=np.float32)
        values = np.asarray(data[f"{variable}_vals"], dtype=np.float32)
        eligible_indices = np.flatnonzero(per_variable_eligible[variable])
        is_retained = np.zeros(locations.shape[0], dtype=bool)
        is_excluded = np.zeros(locations.shape[0], dtype=bool)
        is_retained[eligible_indices[~eligible_is_excluded]] = True
        is_excluded[eligible_indices[eligible_is_excluded]] = True
        retained[f"{variable}_locs"] = locations[is_retained]
        retained[f"{variable}_vals"] = values[is_retained]
        excluded[f"{variable}_locs"] = locations[is_excluded]
        excluded[f"{variable}_vals"] = values[is_excluded]
        excluded[f"{variable}_era5_flat_cells"] = cells[eligible_is_excluded]
        rows.append(
            {
                "timestep": timestep,
                "source": "surface",
                "variable": variable,
                "spatial_domain": "CONUS by native report coordinates",
                "holdout_unit": "ERA5 cell shared across surface variables",
                "n_input_obs": int(locations.shape[0]),
                "n_original_obs": int(cells.size),
                "n_retained_obs": int(is_retained.sum()),
                "n_excluded_obs": int(is_excluded.sum()),
                "n_original_cells": int(np.unique(cells).size),
                "n_excluded_cells": int(
                    np.unique(cells[eligible_is_excluded]).size
                ),
            }
        )
    excluded["heldout_era5_flat_cells"] = excluded_cells
    return retained, excluded, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["aircraft", "surface"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--timesteps", type=Path, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--interface-config", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must lie strictly between 0 and 1")
    with args.timesteps.open(encoding="utf-8") as handle:
        timesteps = [int(value) for value in json.load(handle)["timesteps"]]
    if len(timesteps) != 24 or len(set(timesteps)) != 24:
        raise SystemExit("The paper holdout manifest must contain 24 unique times")

    lat_axis = np.load(args.era5_root / "lat.npy").astype(np.float32)
    lon_axis = np.load(args.era5_root / "lon.npy").astype(np.float32)
    with args.interface_config.open(encoding="utf-8") as handle:
        interface = json.load(handle)
    allowed_source_codes = {
        int(value)
        for value in interface["aircraft_interface"]["madis_data_source_codes"]
    }
    retained_root = args.output_root / "obs"
    excluded_root = args.output_root / "heldout_obs"
    tables_root = args.output_root / "tables"
    retained_root.mkdir(parents=True, exist_ok=True)
    excluded_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    prefix = "madis_aircraft_13var" if args.source == "aircraft" else "madis_metar_surface_13var"
    rows: list[dict] = []
    for timestep in timesteps:
        source_path = args.input_root / "obs" / f"{prefix}_t{timestep:04d}.npz"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        with np.load(source_path, allow_pickle=False) as data:
            split_function = split_aircraft if args.source == "aircraft" else split_surface
            split_args = [
                data,
                timestep,
                lat_axis,
                lon_axis,
                args.seed,
                args.holdout_fraction,
            ]
            if args.source == "aircraft":
                split_args.append(allowed_source_codes)
            retained, excluded, timestep_rows = split_function(*split_args)
            metadata = {
                **load_metadata(data),
                "holdout_protocol": (
                    "aircraft_conus_cell_variable_train80_holdout20"
                    if args.source == "aircraft"
                    else "surface_conus_cell_train80_holdout20"
                ),
                "holdout_fraction": args.holdout_fraction,
                "holdout_seed": args.seed,
                "spatial_domain": {
                    "name": "CONUS",
                    "selection": "native report coordinates before cell assignment",
                    "latitude_degrees_north": [CONUS_BOUNDS[0], CONUS_BOUNDS[1]],
                    "longitude_degrees_east": [CONUS_BOUNDS[2], CONUS_BOUNDS[3]],
                },
                "timestep": timestep,
                "created_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }
            retained["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True)
            )
            excluded["metadata_json"] = np.asarray(
                json.dumps(metadata, sort_keys=True)
            )
        np.savez_compressed(retained_root / source_path.name, **retained)
        np.savez_compressed(excluded_root / source_path.name, **excluded)
        rows.extend(timestep_rows)

    summary_path = tables_root / f"{args.source}_holdout_split_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": args.source,
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "era5_root": str(args.era5_root.resolve()),
        "timesteps_manifest": str(args.timesteps.resolve()),
        "interface_config": str(args.interface_config.resolve()),
        "aircraft_data_source_codes": sorted(allowed_source_codes),
        "timesteps": timesteps,
        "seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "spatial_domain": {
            "name": "CONUS",
            "selection": "native report coordinates before cell assignment",
            "latitude_degrees_north": [CONUS_BOUNDS[0], CONUS_BOUNDS[1]],
            "longitude_degrees_east": [CONUS_BOUNDS[2], CONUS_BOUNDS[3]],
        },
        "n_files": len(timesteps),
        "retained_observations": str(retained_root.resolve()),
        "excluded_observations": str(excluded_root.resolve()),
        "summary_csv": str(summary_path.resolve()),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
