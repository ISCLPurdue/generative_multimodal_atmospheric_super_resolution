#!/usr/bin/env python3
"""Combine matched R+A, R+S, and R+A+S annual RMSE summaries.

Each input must contain one row per evaluation region. The script accepts both
the original analysis column names and the descriptive names emitted by the
release analysis script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = {
    "all13_mean_rmse_change_pct": (
        "all13_mean_rmse_change_pct",
        "all13_mean_pct_delta",
    ),
    "surface3_mean_rmse_change_pct": (
        "surface3_mean_rmse_change_pct",
        "surface3_mean_pct_delta",
    ),
    "aircraft6_mean_rmse_change_pct": (
        "aircraft6_mean_rmse_change_pct",
        "constrained6_mean_pct_delta",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ra-summary", type=Path, required=True)
    parser.add_argument("--rs-summary", type=Path, required=True)
    parser.add_argument("--ras-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def select_metric(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    raise ValueError(f"Missing metric column; expected one of {candidates}")


def load_summary(path: Path, configuration: str) -> pd.DataFrame:
    source = pd.read_csv(path)
    if source["region"].duplicated().any():
        raise ValueError(f"Expected one row per region in {path}")
    result = pd.DataFrame(
        {
            "region": source["region"],
            "configuration": configuration,
            "n_analysis_times": source["n_eval_timesteps"],
        }
    )
    for output_name, candidates in METRICS.items():
        result[output_name] = select_metric(source, candidates)
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(
        [
            load_summary(args.ra_summary, "R+A"),
            load_summary(args.rs_summary, "R+S"),
            load_summary(args.ras_summary, "R+A+S"),
        ],
        ignore_index=True,
    )
    if set(summary["n_analysis_times"]) != {723}:
        raise ValueError("All composition summaries must use 723 analysis times")
    summary.to_csv(args.output_dir / "composition_summary.csv", index=False)

    within_conus = summary[summary["region"].eq("strict_conus")].set_index(
        "configuration"
    )
    rows = []
    labels = {
        "all13_mean_rmse_change_pct": "all13",
        "surface3_mean_rmse_change_pct": "surface-targeted",
        "aircraft6_mean_rmse_change_pct": "aircraft-targeted",
    }
    for column, label in labels.items():
        values = within_conus[column]
        best_single = min(values["R+A"], values["R+S"])
        rows.append(
            {
                "metric": label,
                "R+A": values["R+A"],
                "R+S": values["R+S"],
                "R+A+S": values["R+A+S"],
                "R+A+S_minus_best_single_pp": values["R+A+S"] - best_single,
                "best_single": "R+A" if values["R+A"] <= values["R+S"] else "R+S",
            }
        )
    pd.DataFrame(rows).to_csv(
        args.output_dir / "complementarity_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
