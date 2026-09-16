#!/usr/bin/env python3
"""Redraw the appendix A and S interface-design comparison figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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


def annotation_color(cmap, norm, value: float) -> str:
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


def save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def draw_aircraft(summary: pd.DataFrame) -> tuple[Path, Path]:
    frame = summary.loc[
        summary["region"].eq("strict_conus")
        & summary["modality"].eq("aircraft")
    ].copy()

    reporting_system_labels = {
        "all_qc": "All reporting systems",
        "exclude_tamdar": "All except TAMDAR",
        "selected_reporting_systems": "Selected reporting systems",
    }
    representation_labels = {
        "v1_pointwise": "Individual-report\nresiduals",
        "v2_cell_balanced_pointwise": "Cell-balanced\nindividual residuals",
        "v4_equal_cell": "Equal-cell\nmean residuals",
        "v4c_distance_weighted_cell": "Distance-weighted\ncell-mean residuals",
    }

    frame["row_label"] = frame.apply(
        lambda row: (
            f"{reporting_system_labels[row.source_policy]}\n"
            f"±{row.pressure_window_hpa:d} hPa"
        ),
        axis=1,
    )
    row_order = [
        f"{reporting_system_labels[systems]}\n±{window:d} hPa"
        for systems in ("all_qc", "exclude_tamdar", "selected_reporting_systems")
        for window in (5, 25)
    ]
    column_order = [
        "v1_pointwise",
        "v2_cell_balanced_pointwise",
        "v4_equal_cell",
        "v4c_distance_weighted_cell",
    ]
    matrix = (
        frame.pivot(
            index="row_label",
            columns="operator",
            values="all_13_variables_mean",
        )
        .loc[row_order, column_order]
        .to_numpy(float)
    )

    bound = max(1.0, float(np.nanmax(np.abs(matrix))))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    cmap = plt.get_cmap("RdBu_r")

    figure, axis = plt.subplots(figsize=(12.4, 7.4))
    figure.subplots_adjust(left=0.31, right=0.89, bottom=0.19, top=0.88)
    image = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(
        range(len(column_order)),
        [representation_labels[value] for value in column_order],
    )
    axis.set_yticks(range(len(row_order)), row_order)
    axis.set_xlabel("Residual representation", labelpad=10)
    axis.set_ylabel("Reporting-system selection and pressure window", labelpad=24)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color=annotation_color(cmap, norm, value),
                fontsize=13,
                fontweight="bold",
            )

    colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.055)
    colorbar.set_label("Mean RMSE change relative to R-only conditioning (%)")
    colorbar.ax.tick_params(labelsize=12.5)
    axis.set_title("Aircraft interface-design comparison")

    return save_figure(figure, "figure14_aircraft_interface_design")


def draw_surface_station(summary: pd.DataFrame) -> tuple[Path, Path]:
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
    all_values = frame.loc[representations, "all_13_variables_mean"].to_numpy(float)
    surface_values = frame.loc[representations, "surface_targeted_mean"].to_numpy(float)
    positions = np.arange(len(representations), dtype=float)

    figure, axis = plt.subplots(figsize=(11.2, 5.3))
    figure.subplots_adjust(left=0.31, right=0.96, bottom=0.22, top=0.84)
    axis.axvline(0.0, color="#65727f", linestyle="--", linewidth=1.4, zorder=0)
    axis.grid(axis="x", color="#d8dee4", linewidth=0.9, alpha=0.9, zorder=0)

    axis.scatter(
        all_values,
        positions + 0.13,
        s=105,
        marker="o",
        color="#2c78b7",
        edgecolor="white",
        linewidth=0.9,
        label="All 13 variables",
        zorder=3,
    )
    axis.scatter(
        surface_values,
        positions - 0.13,
        s=115,
        marker="s",
        color="#12988f",
        edgecolor="white",
        linewidth=0.9,
        label="Surface-targeted variables",
        zorder=3,
    )

    for values, offsets in (
        (all_values, positions + 0.13),
        (surface_values, positions - 0.13),
    ):
        for value, vertical_position in zip(values, offsets):
            axis.annotate(
                f"{value:.2f}%",
                (value, vertical_position),
                xytext=(-8, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=12.5,
            )

    axis.set_yticks(positions, labels)
    axis.set_ylim(-0.55, len(positions) - 0.45)
    axis.invert_yaxis()
    axis.set_xlim(min(surface_values) - 2.0, 0.8)
    axis.set_xlabel("Mean RMSE change relative to R-only conditioning (%)", labelpad=9)
    axis.set_ylabel("Residual representation", labelpad=18)
    axis.set_title("Surface-station residual-representation comparison")
    axis.legend(loc="upper right", frameon=True, framealpha=0.96)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    return save_figure(
        figure, "figure15_surface_station_residual_representation"
    )


def main() -> None:
    configure_style()
    summary = pd.read_csv(SUMMARY)
    summary["pressure_window_hpa"] = (
        summary["pressure_window"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(int)
    )
    known_policies = {"all_qc", "exclude_tamdar", "selected_reporting_systems"}
    summary.loc[~summary["source_policy"].isin(known_policies), "source_policy"] = (
        "selected_reporting_systems"
    )
    outputs = [draw_aircraft(summary), draw_surface_station(summary)]
    for pdf_path, png_path in outputs:
        print(pdf_path)
        print(png_path)


if __name__ == "__main__":
    main()
