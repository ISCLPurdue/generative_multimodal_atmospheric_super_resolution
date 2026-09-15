#!/usr/bin/env python3
"""Redraw RMSE-change figures with the manuscript's unified terminology.

The archived composition table contains both competing annual summaries.  This
script deliberately uses ``mean_pct_delta_over_timesteps``: the relative RMSE
change is computed within each matched analysis time and then averaged across
the 723 analysis times.  No posterior samples are recomputed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path("/depot/rmaulik/data/yangxu")
VERSION = Path(__file__).resolve().parents[1]
FIGURES = VERSION / "figures"
TABLES = VERSION / "tables"
SOURCE_TABLE = (
    ROOT
    / "reports/2026/07272026report"
    / "20260727__frozen2019_full723_composition_evidence"
    / "tables/frozen2019_full723_composition_per_variable.csv"
)
COMPOSITION_SUMMARY = (
    ROOT
    / "reports/2026/07272026report"
    / "20260727__frozen2019_full723_composition_evidence"
    / "tables/frozen2019_full723_composition_summary.csv"
)
BOOTSTRAP_SUMMARY = (
    ROOT
    / "reports/2026/07262026report"
    / "20260726__frozen2019_RAS_fullyear2020_analysis"
    / "tables/per_variable_14day_summary_full723.csv"
)

COMPOSITIONS = ("R+A", "R+S", "R+A+S")
COLORS = {"R+A": "#2A6FAD", "R+S": "#229E38", "R+A+S": "#7651A8"}
VARIABLE_ORDER = (
    "t2m",
    "u10",
    "v10",
    "z500",
    "z850",
    "u500",
    "u850",
    "v500",
    "v850",
    "t500",
    "t850",
    "q500",
    "q850",
)


def configure_typography() -> None:
    """Match the Nimbus Sans typography used by the manuscript figures."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12.5,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_values() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_TABLE)
    frame = frame[frame["region"].eq("strict_conus")].copy()
    frame = frame[
        [
            "composition",
            "variable",
            "var_short",
            "mean_pct_delta_over_timesteps",
        ]
    ]
    frame = frame.rename(
        columns={
            "mean_pct_delta_over_timesteps": "mean_paired_relative_rmse_change_pct"
        }
    )
    frame["var_short"] = pd.Categorical(
        frame["var_short"], VARIABLE_ORDER, ordered=True
    )
    frame = frame.sort_values(["var_short", "composition"]).reset_index(drop=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        TABLES / "figure6_per_variable_mean_rmse_change_v155.csv",
        index=False,
    )
    return frame


def draw(frame: pd.DataFrame) -> None:
    y = np.arange(len(VARIABLE_ORDER))
    height = 0.22
    offsets = {"R+A": -height, "R+S": 0.0, "R+A+S": height}

    fig, ax = plt.subplots(figsize=(9.8, 7.8), constrained_layout=True)
    for index in range(len(VARIABLE_ORDER)):
        if index % 2 == 0:
            ax.axhspan(index - 0.5, index + 0.5, color="#F1F4F8", zorder=0)

    for composition in COMPOSITIONS:
        subset = (
            frame[frame["composition"].eq(composition)]
            .set_index("var_short")
            .loc[list(VARIABLE_ORDER)]
        )
        ax.barh(
            y + offsets[composition],
            subset["mean_paired_relative_rmse_change_pct"],
            height=height * 0.86,
            color=COLORS[composition],
            edgecolor="white",
            linewidth=0.45,
            label=composition,
            zorder=3,
        )

    ax.axvline(0.0, color="#596273", linewidth=1.15, linestyle="--")
    ax.grid(axis="x", color="#D7DDE5", linewidth=0.9, alpha=0.9)
    ax.set_yticks(y, [rf"${name}$" for name in VARIABLE_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel(r"Per-variable mean RMSE change, $\Delta_{k,G}^{P}$ (%)")
    ax.set_title("Per-variable RMSE changes over the CONUS domain")
    ax.legend(loc="lower left", frameon=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "figure6_per_variable_mean_rmse_change_v155"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def annotation_color(cmap, norm: Normalize, value: float) -> str:
    """Choose readable annotation text for a heat-map cell."""
    red, green, blue, _ = cmap(norm(value))

    def linearize(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    luminance = (
        0.2126 * linearize(red)
        + 0.7152 * linearize(green)
        + 0.0722 * linearize(blue)
    )
    return "white" if luminance < 0.34 else "black"


def draw_grouped_mean() -> None:
    """Redraw the grouped comparison without changing any numerical values."""
    summary = pd.read_csv(COMPOSITION_SUMMARY)
    strict = summary[summary["region"].eq("strict_conus")]
    groups = {
        "all 13 variables": "all13_mean_pct_delta",
        "surface-targeted": "surface3_mean_pct_delta",
        "aircraft-targeted": "constrained6_mean_pct_delta",
    }
    matrix = np.asarray(
        [
            [
                float(strict[strict["composition"].eq(composition)].iloc[0][column])
                for column in groups.values()
            ]
            for composition in COMPOSITIONS
        ]
    )

    fig, ax = plt.subplots(figsize=(10.4, 4.65), constrained_layout=True)
    cmap = plt.get_cmap("YlGnBu_r")
    norm = Normalize(vmin=-15.0, vmax=0.0)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(3), list(groups), fontsize=12)
    ax.set_yticks(np.arange(3), list(COMPOSITIONS), fontsize=13)
    for row in range(3):
        for column in range(3):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:+.2f}%",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                color=annotation_color(cmap, norm, matrix[row, column]),
            )
    ax.set_title("Grouped RMSE changes over the CONUS domain", fontsize=17, pad=9)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.025)
    colorbar.set_label(
        r"Grouped mean RMSE change, $\overline{\Delta}_{G,\mathcal{K}'}^{P}$ (%)",
        fontsize=12,
    )
    colorbar.ax.tick_params(labelsize=11)
    stem = FIGURES / "figure5_grouped_mean_rmse_change_v155"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def variable_category(name: str) -> tuple[str, str, str]:
    if name in {"t2m", "u10", "v10"}:
        return "Surface-targeted", "#0F8F83", "s"
    if name in {"t500", "t850", "u500", "u850", "v500", "v850"}:
        return "Aircraft-targeted", "#D97706", "D"
    return "Other state variables", "#44546A", "o"


def draw_per_variable_intervals() -> None:
    """Redraw the Appendix intervals with the same archived estimates."""
    data = pd.read_csv(BOOTSTRAP_SUMMARY)
    data = data[data["region"].eq("strict_conus")].copy()
    data = data.set_index("var_short").loc[list(VARIABLE_ORDER)]
    means = data["mean_effect_pct_full723"].astype(float).to_numpy()
    lows = data["ci95_primary_low_pct"].astype(float).to_numpy()
    highs = data["ci95_primary_high_pct"].astype(float).to_numpy()
    y = np.arange(len(VARIABLE_ORDER))

    fig, ax = plt.subplots(figsize=(8.4, 5.7), constrained_layout=True)
    ax.set_xlim(min(lows) - 0.9, max(0.0, max(highs)) + 1.55)
    xmin, xmax = ax.get_xlim()
    ax.axvspan(xmin, min(0.0, xmax), color="#EAF5EF", alpha=0.68, zorder=0)
    if xmax > 0:
        ax.axvspan(max(0.0, xmin), xmax, color="#FBEDEE", alpha=0.62, zorder=0)
    ax.axvline(0.0, color="#596273", linewidth=1.15, linestyle="--", zorder=1)
    ax.set_xlim(xmin, xmax)

    for idx, variable in enumerate(VARIABLE_ORDER):
        _, color, marker = variable_category(variable)
        value = means[idx]
        ax.errorbar(
            value,
            idx,
            xerr=[[value - lows[idx]], [highs[idx] - value]],
            fmt=marker,
            markersize=7.2,
            markerfacecolor="white",
            markeredgewidth=1.7,
            color=color,
            ecolor=color,
            elinewidth=1.9,
            capsize=3.7,
            zorder=3,
        )
        ax.annotate(
            f"{value:+.2f}%",
            (highs[idx], idx),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9.6,
            color="#263238",
        )
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, color="#F2F4F7", alpha=0.52, zorder=0)

    ax.set_yticks(y, [rf"${name}$" for name in VARIABLE_ORDER])
    ax.invert_yaxis()
    ax.set_xlabel(
        r"Per-variable mean RMSE change, $\Delta_{k,G}^{\mathrm{R+A+S}}$ (%)"
    )
    ax.set_title("Per-variable 14-day moving-block intervals over the CONUS domain", pad=23)
    ax.text(
        0.5,
        1.008,
        "Points show per-variable means and bars show 95% intervals",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.8,
        color="#39434D",
    )
    ax.grid(axis="x", color="#D7DDE5", linewidth=0.85, alpha=0.9, zorder=0)
    handles = []
    seen = set()
    for variable in VARIABLE_ORDER:
        label, color, marker = variable_category(variable)
        if label not in seen:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color=color,
                    markerfacecolor="white",
                    markeredgewidth=1.7,
                    linewidth=1.9,
                    label=label,
                )
            )
            seen.add(label)
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#C8CDD4",
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    stem = FIGURES / "appendix_per_variable_mean_rmse_change_v155"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_typography()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    draw_grouped_mean()
    draw(load_values())
    draw_per_variable_intervals()


if __name__ == "__main__":
    main()
