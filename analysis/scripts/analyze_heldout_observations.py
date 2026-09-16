#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_paths import output_path, required_path

ERA5_ROOT = required_path("ERA5_ROOT")
HOLDOUT_TARGETS = required_path("HOLDOUT_CELL_TARGETS_CSV")
OUT = output_path("HELDOUT_ANALYSIS_OUTPUT_ROOT", "heldout_evaluation")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

RA_ROOT = required_path("RA_SAMPLES_ROOT")
RS_ROOT = required_path("RS_SAMPLES_ROOT")
RAS_ROOT = required_path("RAS_SAMPLES_ROOT")
SURFACE_HOLDOUT_ROOT = required_path("SURFACE_HOLDOUT_SAMPLES_ROOT")
AIRCRAFT_HOLDOUT_ROOT = required_path("AIRCRAFT_HOLDOUT_SAMPLES_ROOT")

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential_500",
    "geopotential_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
    "temperature_500",
    "temperature_850",
    "specific_humidity_500",
    "specific_humidity_850",
]
SHORT = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "u_component_of_wind_500": "u500",
    "u_component_of_wind_850": "u850",
    "v_component_of_wind_500": "v500",
    "v_component_of_wind_850": "v850",
    "temperature_500": "t500",
    "temperature_850": "t850",
}
SURFACE_ORDER = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
AIRCRAFT_ORDER = [
    "temperature_500",
    "temperature_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
]
SAMPLE_RE = re.compile(r"_t(\d{4})_e16_s50\.npy$")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 17


def sample_map(root: Path) -> dict[int, Path]:
    mapping = {}
    for path in root.glob("*.npy"):
        match = SAMPLE_RE.search(path.name)
        if match:
            mapping[int(match.group(1))] = path
    return mapping


def load_normalization() -> tuple[np.ndarray, np.ndarray]:
    mean_npz = np.load(ERA5_ROOT / "normalize_mean.npz")
    std_npz = np.load(ERA5_ROOT / "normalize_std.npz")
    mean = np.asarray(
        [float(np.asarray(mean_npz[name]).reshape(-1)[0]) for name in VARIABLES],
        dtype=np.float32,
    )[:, None, None]
    std = np.asarray(
        [float(np.asarray(std_npz[name]).reshape(-1)[0]) for name in VARIABLES],
        dtype=np.float32,
    )[:, None, None]
    return mean, std


def load_ensemble_mean(path: Path, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    array = np.load(path, mmap_mode="r")
    normalized_mean = np.asarray(array, dtype=np.float32).mean(axis=0)
    return normalized_mean * std + mean


def select_values(field: np.ndarray, variable: str, flat_cells: np.ndarray) -> np.ndarray:
    return field[VARIABLES.index(variable)].reshape(-1)[flat_cells]


def attach_predictions(targets: pd.DataFrame) -> pd.DataFrame:
    mean, std = load_normalization()
    maps = {
        "RA": sample_map(RA_ROOT),
        "RS": sample_map(RS_ROOT),
        "RAS": sample_map(RAS_ROOT),
        "surface80": sample_map(SURFACE_HOLDOUT_ROOT),
        "aircraft80": sample_map(AIRCRAFT_HOLDOUT_ROOT),
    }
    timesteps = sorted(int(value) for value in targets["timestep"].unique())
    for name, mapping in maps.items():
        missing = sorted(set(timesteps) - set(mapping))
        if missing:
            raise RuntimeError(f"{name} missing timesteps: {missing}")

    rows = []
    for timestep in timesteps:
        print(f"t{timestep:04d}", flush=True)
        fields = {
            name: load_ensemble_mean(mapping[timestep], mean, std)
            for name, mapping in maps.items()
        }
        subset = targets[targets["timestep"].eq(timestep)]
        for (family, variable), group in subset.groupby(["family", "variable"], sort=False):
            cells = group["flat_cell"].to_numpy(dtype=int)
            target = group["target"].to_numpy(dtype=float)
            if family == "surface":
                methods = {
                    "matched_baseline": ("R+A", fields["RA"]),
                    "heldout80": ("R+A+0.8S", fields["surface80"]),
                    "full_RAS": ("R+A+S", fields["RAS"]),
                }
            else:
                methods = {
                    "matched_baseline": ("R+S", fields["RS"]),
                    "heldout80": ("R+0.8A+S", fields["aircraft80"]),
                    "full_RAS": ("R+A+S", fields["RAS"]),
                }
            for method, (display, field) in methods.items():
                prediction = select_values(field, variable, cells)
                error = prediction - target
                rows.append(
                    {
                        "family": family,
                        "timestep": timestep,
                        "variable": variable,
                        "var_short": SHORT[variable],
                        "method": method,
                        "method_display": display,
                        "n_targets": len(error),
                        "rmse": float(np.sqrt(np.mean(error * error))),
                        "mae": float(np.mean(np.abs(error))),
                    }
                )
    return pd.DataFrame(rows)


def paired_effects(rmse: pd.DataFrame) -> pd.DataFrame:
    key = ["family", "timestep", "variable", "var_short"]
    baseline = rmse[rmse["method"].eq("matched_baseline")][key + ["rmse"]].rename(
        columns={"rmse": "rmse_matched_baseline"}
    )
    effects = rmse.merge(baseline, on=key, how="left")
    effects["pct_change_vs_matched_baseline"] = (
        100.0
        * (effects["rmse"] - effects["rmse_matched_baseline"])
        / effects["rmse_matched_baseline"]
    )
    return effects


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    replicates = values[indices].mean(axis=1)
    return tuple(np.quantile(replicates, [0.025, 0.975]))


def summarize_variables(effects: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    selected = effects[effects["method"].eq("heldout80")]
    for keys, group in selected.groupby(
        ["family", "variable", "var_short", "method", "method_display"], sort=False
    ):
        values = group["pct_change_vs_matched_baseline"].to_numpy(dtype=float)
        low, high = bootstrap_mean(values, rng)
        rows.append(
            {
                "family": keys[0],
                "variable": keys[1],
                "var_short": keys[2],
                "method": keys[3],
                "method_display": keys[4],
                "n_timesteps": len(values),
                "mean_effect_pct": float(values.mean()),
                "ci95_low_pct": low,
                "ci95_high_pct": high,
                "improved_timestep_fraction": float(np.mean(values < 0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_families(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestep_effects = (
        effects[effects["method"].eq("heldout80")]
        .groupby(["family", "method", "method_display", "timestep"], as_index=False)[
            "pct_change_vs_matched_baseline"
        ]
        .mean()
        .rename(columns={"pct_change_vs_matched_baseline": "timestep_mean_effect_pct"})
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for keys, group in timestep_effects.groupby(
        ["family", "method", "method_display"], sort=False
    ):
        values = group["timestep_mean_effect_pct"].to_numpy(dtype=float)
        low, high = bootstrap_mean(values, rng)
        rows.append(
            {
                "family": keys[0],
                "method": keys[1],
                "method_display": keys[2],
                "n_timesteps": len(values),
                "mean_effect_pct": float(values.mean()),
                "ci95_low_pct": low,
                "ci95_high_pct": high,
                "improved_timestep_fraction": float(np.mean(values < 0)),
            }
        )
    return pd.DataFrame(rows), timestep_effects


def make_figure(family_summary: pd.DataFrame, variable_summary: pd.DataFrame) -> Path:
    output = FIGURES / "matched_marginal_holdout_summary.png"
    navy, orange, teal, green = "#172A46", "#E67E22", "#0F8B8D", "#27966F"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        gridspec_kw={"width_ratios": [0.78, 1.52]},
        constrained_layout=True,
    )

    headline = family_summary[family_summary["method"].eq("heldout80")].set_index("family")
    families = ["surface", "aircraft"]
    means = np.asarray([headline.loc[name, "mean_effect_pct"] for name in families])
    lows = np.asarray([headline.loc[name, "ci95_low_pct"] for name in families])
    highs = np.asarray([headline.loc[name, "ci95_high_pct"] for name in families])
    colors = [teal, orange]
    axes[0].bar(
        np.arange(2),
        means,
        color=colors,
        width=0.58,
        yerr=np.vstack([means - lows, highs - means]),
        capsize=5,
        edgecolor="white",
    )
    axes[0].axhline(0, color="#586574", linestyle="--", linewidth=1.2)
    axes[0].set_xticks([0, 1], ["Excluded S", "Excluded A"])
    axes[0].set_ylabel("Mean paired RMSE change (%)")
    axes[0].set_title("(a) Matched marginal holdout", loc="left", weight="bold", color=navy)
    axes[0].grid(axis="y", alpha=0.22)
    for index, value in enumerate(means):
        axes[0].text(index, value - 0.7, f"{value:.2f}%", ha="center", va="top", color="white", weight="bold")

    order = SURFACE_ORDER + AIRCRAFT_ORDER
    selected = variable_summary[variable_summary["method"].eq("heldout80")].set_index("variable")
    y = np.arange(len(order))
    for index, variable in enumerate(order):
        row = selected.loc[variable]
        value = row["mean_effect_pct"]
        axes[1].errorbar(
            value,
            index,
            xerr=[[value - row["ci95_low_pct"]], [row["ci95_high_pct"] - value]],
            fmt="o",
            color=teal if variable in SURFACE_ORDER else orange,
            ecolor=teal if variable in SURFACE_ORDER else orange,
            capsize=3,
            markersize=6,
        )
        if index % 2 == 0:
            axes[1].axhspan(index - 0.5, index + 0.5, color="#F3F6F9", zorder=-5)
    axes[1].axvline(0, color="#586574", linestyle="--", linewidth=1.2)
    axes[1].set_yticks(y, [SHORT[name] for name in order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean paired RMSE change at excluded targets (%)")
    axes[1].set_title("(b) Variable-level effects", loc="left", weight="bold", color=navy)
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].scatter([], [], color=teal, label="Surface-station targets")
    axes[1].scatter([], [], color=orange, label="Aircraft targets")
    axes[1].legend(frameon=True, loc="lower left")

    fig.suptitle(
        "Interfaces selected with 2019 data improve 2020 observations excluded from conditioning",
        fontsize=15,
        weight="bold",
        color=navy,
    )
    fig.savefig(output, dpi=260)
    plt.close(fig)
    return output


def write_readme(family_summary: pd.DataFrame, variable_summary: pd.DataFrame, figure: Path) -> None:
    heldout = family_summary[family_summary["method"].eq("heldout80")].set_index("family")
    surface = heldout.loc["surface"]
    aircraft = heldout.loc["aircraft"]
    text = f"""# Matched-marginal evaluation at observations excluded from conditioning

This analysis summarizes the completed 2020 holdout runs.
It replaces the earlier R-only pooled comparison with the matched marginal
baselines required by the manuscript design:

- excluded S: `R+A+0.8S` versus `R+A`;
- excluded A: `R+0.8A+S` versus `R+S`.

For each of the 24 prespecified 2020 diagnostic cases, RMSE is computed at the
same held-out ERA5 cell-variable targets. Percentage changes are formed within
each timestep and variable, then averaged with equal variable and timestep
weight. The 95% intervals use {BOOTSTRAP_REPLICATES:,} paired-timestep bootstrap
replicates with seed {BOOTSTRAP_SEED}.

## Headline effects

- Excluded surface-station targets: `{surface.mean_effect_pct:+.3f}%`
  (95% CI `{surface.ci95_low_pct:+.3f}` to `{surface.ci95_high_pct:+.3f}%`);
  improvement in `{100 * surface.improved_timestep_fraction:.1f}%` of timesteps.
- Excluded aircraft targets: `{aircraft.mean_effect_pct:+.3f}%`
  (95% CI `{aircraft.ci95_low_pct:+.3f}` to `{aircraft.ci95_high_pct:+.3f}%`);
  improvement in `{100 * aircraft.improved_timestep_fraction:.1f}%` of timesteps.

These are interpolation diagnostics over one fixed 80/20 cell split and one
ensemble seed, not independent-network or long-gap temporal validation.

## Outputs

- `{TABLES / 'rmse_by_timestep_variable.csv'}`
- `{TABLES / 'paired_effects_by_timestep_variable.csv'}`
- `{TABLES / 'heldout_variable_summary.csv'}`
- `{TABLES / 'heldout_family_summary.csv'}`
- `{TABLES / 'matched_marginal_timestep_effects.csv'}`
- `{figure}`
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(HOLDOUT_TARGETS)
    rmse = attach_predictions(targets)
    effects = paired_effects(rmse)
    variable_summary = summarize_variables(effects)
    family_summary, timestep_effects = summarize_families(effects)
    figure = make_figure(family_summary, variable_summary)

    rmse.to_csv(TABLES / "rmse_by_timestep_variable.csv", index=False)
    effects.to_csv(TABLES / "paired_effects_by_timestep_variable.csv", index=False)
    variable_summary.to_csv(TABLES / "heldout_variable_summary.csv", index=False)
    family_summary.to_csv(TABLES / "heldout_family_summary.csv", index=False)
    timestep_effects.to_csv(TABLES / "matched_marginal_timestep_effects.csv", index=False)
    write_readme(family_summary, variable_summary, figure)
    print(
        json.dumps(
            {
                "report": str(OUT),
                "family_summary": family_summary.to_dict(orient="records"),
                "figure": str(figure),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
