#!/usr/bin/env python3
"""Build cell-mean targets for the paper's held-out-observation evaluation."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


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
SHORT_NAMES = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "temperature_500": "t500",
    "temperature_850": "t850",
    "u_component_of_wind_500": "u500",
    "u_component_of_wind_850": "u850",
    "v_component_of_wind_500": "v500",
    "v_component_of_wind_850": "v850",
}
FILE_RE = re.compile(r"_t(\d{4})\.npz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft-excluded-root",
        type=Path,
        required=True,
        help="Aircraft split's heldout_obs directory.",
    )
    parser.add_argument(
        "--surface-excluded-root",
        type=Path,
        required=True,
        help="Surface split's heldout_obs directory.",
    )
    parser.add_argument(
        "--spatial-domain",
        required=True,
        help="Domain already enforced when the retained/excluded splits were created.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def timestep_from_path(path: Path) -> int:
    match = FILE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot read timestep from {path}")
    return int(match.group(1))


def cell_mean_rows(
    path: Path, family: str, variables: list[str], spatial_domain: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timestep = timestep_from_path(path)
    with np.load(path, allow_pickle=False) as data:
        for variable in variables:
            values_key = f"{variable}_vals"
            cells_key = f"{variable}_era5_flat_cells"
            if values_key not in data.files or cells_key not in data.files:
                raise KeyError(f"{path} is missing {values_key} or {cells_key}")
            values = np.asarray(data[values_key], dtype=np.float64)
            cells = np.asarray(data[cells_key], dtype=np.int64)
            if values.shape != cells.shape:
                raise ValueError(
                    f"{path}: {values_key} shape {values.shape} does not match "
                    f"{cells_key} shape {cells.shape}"
                )
            finite = np.isfinite(values) & (cells >= 0)
            values = values[finite]
            cells = cells[finite]
            for cell in np.unique(cells):
                selected = values[cells == cell]
                rows.append(
                    {
                        "family": family,
                        "spatial_domain": spatial_domain,
                        "timestep": timestep,
                        "variable": variable,
                        "var_short": SHORT_NAMES[variable],
                        "flat_cell": int(cell),
                        "target": float(selected.mean()),
                        "n_obs_in_target": int(selected.size),
                    }
                )
    return rows


def collect(
    root: Path,
    family: str,
    variables: list[str],
    spatial_domain: str,
) -> list[dict[str, object]]:
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ files found under {root}")
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(cell_mean_rows(path, family, variables, spatial_domain))
    return rows


def main() -> None:
    args = parse_args()
    rows = collect(
        args.surface_excluded_root,
        "surface",
        SURFACE_VARIABLES,
        args.spatial_domain,
    ) + collect(
        args.aircraft_excluded_root,
        "aircraft",
        AIRCRAFT_VARIABLES,
        args.spatial_domain,
    )
    rows.sort(
        key=lambda row: (
            int(row["timestep"]),
            str(row["family"]),
            str(row["variable"]),
            int(row["flat_cell"]),
        )
    )
    if not rows:
        raise RuntimeError("No held-out cell targets were produced")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} held-out cell targets to {args.output_csv}")


if __name__ == "__main__":
    main()
