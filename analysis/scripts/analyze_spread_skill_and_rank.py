#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_paths import output_path, required_path

ERA5_ROOT = required_path("ERA5_ROOT")
TIMESTEP_MANIFEST = required_path("EVALUATION_TIMESTEP_MANIFEST")
R_ROOT = required_path("R_ONLY_SAMPLES_ROOT")
RAS_ROOT = required_path("RAS_SAMPLES_ROOT")
OUT = output_path("DISPERSION_ANALYSIS_OUTPUT_ROOT", "dispersion_diagnostics")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

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
    "geopotential_500": "z500",
    "geopotential_850": "z850",
    "u_component_of_wind_500": "u500",
    "u_component_of_wind_850": "u850",
    "v_component_of_wind_500": "v500",
    "v_component_of_wind_850": "v850",
    "temperature_500": "t500",
    "temperature_850": "t850",
    "specific_humidity_500": "q500",
    "specific_humidity_850": "q850",
}
SURFACE = {
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
}
AIRCRAFT = {
    "temperature_500",
    "temperature_850",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
}
GROUPS = {
    "all_13_variables": VARIABLES,
    "surface_targeted_variables": [v for v in VARIABLES if v in SURFACE],
    "aircraft_targeted_variables": [v for v in VARIABLES if v in AIRCRAFT],
}
GROUP_LABELS = {
    "all_13_variables": "All 13 variables",
    "surface_targeted_variables": "Surface-targeted",
    "aircraft_targeted_variables": "Aircraft-targeted",
}
PROTOCOLS = {
    "R": R_ROOT,
    "R+A+S": RAS_ROOT,
}
COLORS = {"R": "#2B6EA6", "R+A+S": "#1B8E72"}
N_MEMBERS = 16
N_RANKS = N_MEMBERS + 1
FINITE_ENSEMBLE_FACTOR = np.sqrt((N_MEMBERS + 1.0) / N_MEMBERS)


def load_timesteps() -> list[int]:
    payload = json.loads(TIMESTEP_MANIFEST.read_text())
    timesteps = [int(value) for value in payload["timesteps"]]
    if len(timesteps) != 723 or len(set(timesteps)) != 723:
        raise ValueError(f"Expected 723 unique timesteps, found {len(timesteps)}")
    return timesteps


def load_normalization() -> tuple[np.ndarray, np.ndarray]:
    means = np.load(ERA5_ROOT / "normalize_mean.npz")
    stds = np.load(ERA5_ROOT / "normalize_std.npz")
    mean = np.asarray(
        [float(np.asarray(means[variable]).reshape(-1)[0]) for variable in VARIABLES],
        dtype=np.float32,
    )
    std = np.asarray(
        [float(np.asarray(stds[variable]).reshape(-1)[0]) for variable in VARIABLES],
        dtype=np.float32,
    )
    return mean[:, None, None], std[:, None, None]


def load_truth(timestep: int) -> np.ndarray:
    with h5py.File(ERA5_ROOT / "test" / f"2020_{timestep:04d}.h5", "r") as handle:
        return np.stack(
            [np.asarray(handle["input"][variable], dtype=np.float32) for variable in VARIABLES],
            axis=0,
        )


def strict_conus_mask() -> np.ndarray:
    lat = np.load(ERA5_ROOT / "lat.npy").astype(float)
    lon = np.load(ERA5_ROOT / "lon.npy").astype(float)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return ((lat >= 24.0) & (lat <= 50.0))[:, None] & ((lon >= -125.0) & (lon <= -66.0))[None, :]


def sample_path(protocol: str, timestep: int) -> Path:
    matches = sorted(PROTOCOLS[protocol].glob(f"*_t{timestep:04d}_e16_s50.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {protocol} sample file for timestep {timestep}, found {len(matches)}"
        )
    return matches[0]


def members_physical(
    protocol: str,
    timestep: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    path = sample_path(protocol, timestep)
    if not path.exists():
        raise FileNotFoundError(path)
    normalized = np.load(path, mmap_mode="r")
    if normalized.shape != (N_MEMBERS, 13, 128, 256):
        raise ValueError(f"Unexpected shape {normalized.shape}: {path}")
    return np.asarray(normalized, dtype=np.float32) * std[None] + mean[None]


def tie_safe_ranks(samples: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    less = np.sum(samples < truth[None], axis=0)
    equal = np.sum(samples == truth[None], axis=0)
    tie_offset = np.floor(rng.random(truth.size) * (equal + 1)).astype(np.int16)
    ranks = less.astype(np.int16) + tie_offset
    if ranks.min() < 0 or ranks.max() >= N_RANKS:
        raise ValueError("Rank outside the expected 0..16 range")
    return ranks


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    mean, std = load_normalization()
    mask = strict_conus_mask()
    timesteps = load_timesteps()
    rng = np.random.default_rng(20260801)

    rmse_sq_sum = np.zeros((len(PROTOCOLS), len(VARIABLES)), dtype=np.float64)
    spread_var_sum = np.zeros_like(rmse_sq_sum)
    rank_counts = np.zeros((len(PROTOCOLS), len(VARIABLES), N_RANKS), dtype=np.int64)
    protocol_names = list(PROTOCOLS)

    for count, timestep in enumerate(timesteps, start=1):
        if count == 1 or count % 25 == 0 or count == len(timesteps):
            print(f"[{count:03d}/{len(timesteps)}] t{timestep:04d}", flush=True)
        truth = load_truth(timestep)
        for protocol_index, protocol in enumerate(protocol_names):
            ensemble = members_physical(protocol, timestep, mean, std)
            ensemble_mean = ensemble.mean(axis=0)
            ensemble_std = ensemble.std(axis=0, ddof=1)
            for channel, variable in enumerate(VARIABLES):
                y = truth[channel][mask]
                samples = ensemble[:, channel][:, mask]
                error = ensemble_mean[channel][mask] - y
                sigma = ensemble_std[channel][mask]
                rmse_sq_sum[protocol_index, channel] += float(np.mean(error * error))
                spread_var_sum[protocol_index, channel] += float(np.mean(sigma * sigma))
                ranks = tie_safe_ranks(samples, y, rng)
                rank_counts[protocol_index, channel] += np.bincount(ranks, minlength=N_RANKS)
            del ensemble, ensemble_mean, ensemble_std

    spread_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for protocol_index, protocol in enumerate(protocol_names):
        for channel, variable in enumerate(VARIABLES):
            rmse = float(np.sqrt(rmse_sq_sum[protocol_index, channel] / len(timesteps)))
            spread = float(np.sqrt(spread_var_sum[protocol_index, channel] / len(timesteps)))
            raw_ratio = spread / rmse
            corrected_ratio = FINITE_ENSEMBLE_FACTOR * raw_ratio
            spread_rows.append(
                {
                    "protocol": protocol,
                    "region": "strict_conus",
                    "variable": variable,
                    "variable_short": SHORT[variable],
                    "n_timesteps": len(timesteps),
                    "n_members": N_MEMBERS,
                    "ensemble_mean_rmse": rmse,
                    "rms_ensemble_spread": spread,
                    "raw_spread_skill_ratio": raw_ratio,
                    "finite_ensemble_corrected_spread_skill_ratio": corrected_ratio,
                }
            )
            counts = rank_counts[protocol_index, channel]
            frequency = counts / counts.sum()
            for rank in range(N_RANKS):
                rank_rows.append(
                    {
                        "protocol": protocol,
                        "region": "strict_conus",
                        "variable": variable,
                        "variable_short": SHORT[variable],
                        "rank": rank,
                        "count": int(counts[rank]),
                        "frequency": float(frequency[rank]),
                        "frequency_relative_to_uniform": float(frequency[rank] * N_RANKS),
                    }
                )
    return pd.DataFrame(spread_rows), pd.DataFrame(rank_rows)


def summarize_groups(
    spread: pd.DataFrame,
    ranks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spread_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for protocol in PROTOCOLS:
        for group, variables in GROUPS.items():
            subset = spread[spread["protocol"].eq(protocol) & spread["variable"].isin(variables)]
            ratios = subset["finite_ensemble_corrected_spread_skill_ratio"].to_numpy(float)
            spread_rows.append(
                {
                    "protocol": protocol,
                    "region": "strict_conus",
                    "evaluation_group": group,
                    "n_variables": len(variables),
                    "mean_corrected_spread_skill_ratio": float(np.mean(ratios)),
                    "min_corrected_spread_skill_ratio": float(np.min(ratios)),
                    "max_corrected_spread_skill_ratio": float(np.max(ratios)),
                }
            )
            rank_subset = ranks[ranks["protocol"].eq(protocol) & ranks["variable"].isin(variables)]
            grouped = rank_subset.groupby("rank", as_index=False)["count"].sum()
            grouped["frequency"] = grouped["count"] / grouped["count"].sum()
            grouped["frequency_relative_to_uniform"] = grouped["frequency"] * N_RANKS
            for row in grouped.itertuples(index=False):
                rank_rows.append(
                    {
                        "protocol": protocol,
                        "region": "strict_conus",
                        "evaluation_group": group,
                        "rank": int(row.rank),
                        "count": int(row.count),
                        "frequency": float(row.frequency),
                        "frequency_relative_to_uniform": float(row.frequency_relative_to_uniform),
                    }
                )
    return pd.DataFrame(spread_rows), pd.DataFrame(rank_rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def plot_main(spread_groups: pd.DataFrame, rank_groups: pd.DataFrame) -> None:
    configure_plotting()
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)

    group_order = list(GROUPS)
    x = np.arange(len(group_order))
    width = 0.34
    for offset, protocol in [(-width / 2, "R"), (width / 2, "R+A+S")]:
        subset = spread_groups[spread_groups["protocol"].eq(protocol)].set_index("evaluation_group")
        axes[0].bar(
            x + offset,
            subset.loc[group_order, "mean_corrected_spread_skill_ratio"],
            width,
            color=COLORS[protocol],
            label=protocol,
        )
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1.2, label="Calibrated reference")
    axes[0].set_xticks(x, ["All 13", "Surface", "Aircraft"], rotation=12)
    axes[0].set_ylabel("Finite-ensemble-corrected spread-skill ratio")
    axes[0].set_title("(a) Dispersion relative to ensemble-mean error")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", color="#D9DEE3", linewidth=0.7, alpha=0.8)
    axes[0].legend(frameon=True)

    all_ranks = rank_groups[rank_groups["evaluation_group"].eq("all_13_variables")]
    for protocol in PROTOCOLS:
        subset = all_ranks[all_ranks["protocol"].eq(protocol)].sort_values("rank")
        axes[1].plot(
            subset["rank"],
            subset["frequency_relative_to_uniform"],
            color=COLORS[protocol],
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=protocol,
        )
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1.2, label="Uniform reference")
    axes[1].set_xticks([0, 4, 8, 12, 16])
    axes[1].set_xlabel("Verification rank among 16 ensemble members")
    axes[1].set_ylabel("Rank frequency / uniform frequency")
    axes[1].set_title("(b) All-variable rank histogram")
    axes[1].grid(color="#D9DEE3", linewidth=0.7, alpha=0.8)
    axes[1].legend(frameon=True)

    figure.suptitle("Full-year 2020 ensemble-dispersion diagnostics within CONUS", fontweight="bold")
    figure.savefig(FIGURES / "spread_skill_rank_histogram_2020.png", dpi=300)
    figure.savefig(FIGURES / "spread_skill_rank_histogram_2020.pdf")
    plt.close(figure)


def plot_group_rank_histograms(rank_groups: pd.DataFrame) -> None:
    configure_plotting()
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True, sharey=True)
    for axis, group in zip(axes, GROUPS):
        for protocol in PROTOCOLS:
            subset = rank_groups[
                rank_groups["protocol"].eq(protocol)
                & rank_groups["evaluation_group"].eq(group)
            ].sort_values("rank")
            axis.plot(
                subset["rank"],
                subset["frequency_relative_to_uniform"],
                color=COLORS[protocol],
                linewidth=1.9,
                marker="o",
                markersize=3.5,
                label=protocol,
            )
        axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1.1)
        axis.set_xticks([0, 4, 8, 12, 16])
        axis.set_xlabel("Verification rank")
        axis.set_title(GROUP_LABELS[group])
        axis.grid(color="#D9DEE3", linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel("Rank frequency / uniform frequency")
    axes[-1].legend(frameon=True)
    figure.suptitle("Rank histograms by evaluation group within CONUS", fontweight="bold")
    figure.savefig(FIGURES / "rank_histograms_by_group_2020.png", dpi=300)
    figure.savefig(FIGURES / "rank_histograms_by_group_2020.pdf")
    plt.close(figure)


def write_readme(spread_groups: pd.DataFrame) -> None:
    lines = [
        "# Full-Year 2020 Spread-Skill and Rank-Histogram Diagnostics",
        "",
        "This CPU-only analysis compares the 16-member R-only and R+A+S ensembles selected using 2019 data over all 723 matched 2020 cases within CONUS.",
        "",
        "## Definitions",
        "",
        "- The raw spread-skill ratio is RMS ensemble spread divided by ensemble-mean RMSE.",
        "- The reported ratio multiplies spread by sqrt((N+1)/N), with N=16, so exchangeable finite ensembles have reference value one.",
        "- Rank histograms use 17 bins for 16 members. Exact ties, if any, are broken uniformly with deterministic seed 20260801.",
        "- Rank frequencies are pooled within each named variable group. Spatial and temporal dependence make the histogram a shape diagnostic, not an independent-count significance test.",
        "",
        "## Grouped corrected spread-skill ratios",
        "",
        "| Evaluation group | R | R+A+S |",
        "|---|---:|---:|",
    ]
    indexed = spread_groups.set_index(["evaluation_group", "protocol"])
    for group in GROUPS:
        lines.append(
            f"| {GROUP_LABELS[group]} | "
            f"{indexed.loc[(group, 'R'), 'mean_corrected_spread_skill_ratio']:.3f} | "
            f"{indexed.loc[(group, 'R+A+S'), 'mean_corrected_spread_skill_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Ratios below one and U-shaped rank histograms both diagnose underdispersion. These diagnostics complement, rather than replace, CRPS and moving-block confidence intervals.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    spread, ranks = evaluate()
    spread_groups, rank_groups = summarize_groups(spread, ranks)
    spread.to_csv(TABLES / "spread_skill_by_variable.csv", index=False)
    spread_groups.to_csv(TABLES / "spread_skill_by_group.csv", index=False)
    ranks.to_csv(TABLES / "rank_histogram_by_variable.csv", index=False)
    rank_groups.to_csv(TABLES / "rank_histogram_by_group.csv", index=False)
    plot_main(spread_groups, rank_groups)
    plot_group_rank_histograms(rank_groups)
    write_readme(spread_groups)
    manifest = {
        "generated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis": "strict-CONUS spread-skill ratios and rank histograms",
        "timestep_manifest": str(TIMESTEP_MANIFEST),
        "n_timesteps": 723,
        "n_members": N_MEMBERS,
        "finite_ensemble_spread_correction": "sqrt((N+1)/N)",
        "rank_tie_seed": 20260801,
        "protocol_roots": {name: str(root) for name, root in PROTOCOLS.items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(spread_groups.to_string(index=False))


if __name__ == "__main__":
    main()
