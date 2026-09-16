#!/usr/bin/env python3
"""Validate a retained/excluded aircraft split against its source files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from create_2020_holdout_split import (
    AIRCRAFT_VARIABLES,
    DEFAULT_INTERFACE,
    DEFAULT_TIMESTEPS,
    conus_mask,
    nearest_era5_flat_cells,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--timesteps", type=Path, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--interface-config", type=Path, default=DEFAULT_INTERFACE)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    args = parse_args()
    timesteps = json.loads(args.timesteps.read_text(encoding="utf-8"))["timesteps"]
    interface = json.loads(args.interface_config.read_text(encoding="utf-8"))
    allowed_codes = set(
        int(value)
        for value in interface["aircraft_interface"]["madis_data_source_codes"]
    )
    lat_axis = np.load(args.era5_root / "lat.npy")
    lon_axis = np.load(args.era5_root / "lon.npy")
    rows: list[dict[str, object]] = []

    for timestep in timesteps:
        name = f"madis_aircraft_13var_t{int(timestep):04d}.npz"
        paths = {
            "source": args.input_root / "obs" / name,
            "retained": args.split_root / "obs" / name,
            "excluded": args.split_root / "heldout_obs" / name,
        }
        with (
            np.load(paths["source"], allow_pickle=False) as source,
            np.load(paths["retained"], allow_pickle=False) as retained,
            np.load(paths["excluded"], allow_pickle=False) as excluded,
        ):
            for variable in AIRCRAFT_VARIABLES:
                source_locations = np.asarray(source[f"{variable}_locs"])
                source_codes = np.asarray(source[f"{variable}_data_source"])
                source_products = np.asarray(source[f"{variable}_source_product"])
                eligible = (
                    conus_mask(source_locations)
                    & np.isin(source_codes, sorted(allowed_codes))
                    & (source_products == "acars")
                )
                eligible_cells = nearest_era5_flat_cells(
                    source_locations[eligible], lat_axis, lon_axis
                )
                retained_locations = np.asarray(retained[f"{variable}_locs"])
                excluded_locations = np.asarray(excluded[f"{variable}_locs"])
                retained_cells = nearest_era5_flat_cells(
                    retained_locations, lat_axis, lon_axis
                )
                excluded_cells = nearest_era5_flat_cells(
                    excluded_locations, lat_axis, lon_axis
                )
                stored_excluded_cells = np.asarray(
                    excluded[f"{variable}_era5_flat_cells"], dtype=np.int64
                )

                label = f"t{int(timestep):04d} {variable}"
                require(
                    conus_mask(retained_locations).all(),
                    f"{label}: retained reports outside CONUS",
                )
                require(
                    conus_mask(excluded_locations).all(),
                    f"{label}: excluded reports outside CONUS",
                )
                require(
                    retained_locations.shape[0] + excluded_locations.shape[0]
                    == int(eligible.sum()),
                    f"{label}: split counts do not partition eligible reports",
                )
                require(
                    not np.intersect1d(retained_cells, excluded_cells).size,
                    f"{label}: retained and excluded ERA5 cells overlap",
                )
                require(
                    np.array_equal(
                        np.union1d(retained_cells, excluded_cells),
                        np.unique(eligible_cells),
                    ),
                    f"{label}: split cell union differs from source cells",
                )
                require(
                    np.array_equal(excluded_cells, stored_excluded_cells),
                    f"{label}: stored excluded-cell indices are inconsistent",
                )
                expected = max(
                    1,
                    int(round(args.holdout_fraction * np.unique(eligible_cells).size)),
                )
                require(
                    np.unique(excluded_cells).size == expected,
                    f"{label}: excluded-cell count differs from requested fraction",
                )
                rows.append(
                    {
                        "timestep": int(timestep),
                        "variable": variable,
                        "n_eligible_reports": int(eligible.sum()),
                        "n_eligible_cells": int(np.unique(eligible_cells).size),
                        "n_retained_reports": int(retained_locations.shape[0]),
                        "n_excluded_reports": int(excluded_locations.shape[0]),
                        "n_excluded_cells": int(np.unique(excluded_cells).size),
                    }
                )

    result = {
        "status": "pass",
        "spatial_domain": "CONUS by native report coordinates",
        "n_analysis_times": len(set(int(value) for value in timesteps)),
        "n_time_variable_checks": len(rows),
        "n_eligible_reports": int(sum(row["n_eligible_reports"] for row in rows)),
        "n_retained_reports": int(sum(row["n_retained_reports"] for row in rows)),
        "n_excluded_reports": int(sum(row["n_excluded_reports"] for row in rows)),
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({**result, "checks": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        csv_path = args.output_json.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
