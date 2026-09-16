#!/usr/bin/env python3
"""Draw Figure 15 as a horizontal grouped bar chart."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_paths import required_path

VERSION_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = VERSION_ROOT / "figures"
SUMMARY = required_path("INTERFACE_DESIGN_SUMMARY_CSV")


def configure_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": False,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    configure_style()
    summary = pd.read_csv(SUMMARY)
    frame = summary.loc[
        summary["region"].eq("strict_conus")
        & summary["modality"].eq("surface_station")
    ].set_index("operator")

    representations = [
        "v4_equal_cell",
        "v2_cell_balanced_pointwise",
        "v1_pointwise",
    ]
    labels = [
        "Equal-cell mean residuals",
        "Cell-balanced individual residuals",
        "Individual-report residuals",
    ]
    all_values = frame.loc[
        representations, "all_13_variables_mean"
    ].to_numpy(float)
    surface_values = frame.loc[
        representations, "surface_targeted_mean"
    ].to_numpy(float)

    positions = np.arange(len(representations), dtype=float)
    height = 0.29

    figure, axis = plt.subplots(figsize=(11.2, 5.4))
    figure.subplots_adjust(left=0.31, right=0.96, bottom=0.20, top=0.76)
    axis.axvline(0.0, color="#65727f", linestyle="--", linewidth=1.4, zorder=1)
    axis.grid(axis="x", color="#d8dee4", linewidth=0.9, alpha=0.9, zorder=0)

    bars_surface = axis.barh(
        positions - height / 2,
        surface_values,
        height=height,
        color="#12988f",
        edgecolor="white",
        linewidth=0.9,
        label="Surface-targeted variables",
        zorder=2,
    )
    bars_all = axis.barh(
        positions + height / 2,
        all_values,
        height=height,
        color="#2c78b7",
        edgecolor="white",
        linewidth=0.9,
        label="All 13 variables",
        zorder=2,
    )

    for bars, values in (
        (bars_surface, surface_values),
        (bars_all, all_values),
    ):
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.2f}%",
                (value, bar.get_y() + bar.get_height() / 2),
                xytext=(-7, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=12.5,
            )

    axis.set_yticks(positions, labels)
    axis.set_ylim(-0.55, len(positions) - 0.45)
    axis.invert_yaxis()
    axis.set_xlim(min(surface_values) - 2.0, 0.8)
    axis.set_xlabel(
        "Mean RMSE change relative to R-only conditioning (%)", labelpad=9
    )
    axis.set_ylabel("Residual representation", labelpad=18)
    figure.suptitle(
        "Surface-station residual-representation comparison",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        framealpha=0.96,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = "figure15_surface_station_residual_representation_grouped_bar"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
