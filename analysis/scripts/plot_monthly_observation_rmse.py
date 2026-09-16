#!/usr/bin/env python3
"""Redraw the appendix monthly observation--ERA5 RMSE figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_paths import required_path

VERSION_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = VERSION_ROOT / "figures"

METAR_MONTHLY = required_path("METAR_MONTHLY_DIAGNOSTICS_CSV")
ABO_TIMESTEP = required_path("AIRCRAFT_TIMESTEP_DIAGNOSTICS_CSV")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": True,
            "axes.formatter.use_mathtext": False,
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "pdf.fonttype": 42,
        }
    )


def annotate_heatmap(axis: plt.Axes, image, matrix: np.ndarray) -> None:
    """Add readable two-decimal annotations to a heatmap."""
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            red, green, blue, _ = image.cmap(image.norm(value))

            def linearize(channel: float) -> float:
                if channel <= 0.04045:
                    return channel / 12.92
                return ((channel + 0.055) / 1.055) ** 2.4

            luminance = (
                0.2126 * linearize(red)
                + 0.7152 * linearize(green)
                + 0.0722 * linearize(blue)
            )
            color = "white" if luminance < 0.34 else "#202A33"
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=11.5,
            )


def add_heatmap(
    figure: plt.Figure,
    axis: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    title: str,
    colorbar_label: str,
) -> None:
    image = axis.imshow(matrix, cmap="viridis", aspect="auto")
    annotate_heatmap(axis, image, matrix)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_xticks(range(12), [str(month) for month in range(1, 13)])
    axis.set_title(title, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, pad=0.012)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(labelsize=12.5)


def save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def plot_surface_station_monthly() -> tuple[Path, Path]:
    monthly = pd.read_csv(METAR_MONTHLY)
    monthly = monthly.loc[monthly["region"].eq("strict_conus")]

    def values(variables: list[str]) -> np.ndarray:
        return (
            monthly.pivot_table(
                index="variable", columns="month", values="rmse", aggfunc="mean"
            )
            .loc[variables, range(1, 13)]
            .to_numpy()
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.2, 6.0),
        gridspec_kw={"height_ratios": [1, 2]},
        layout="constrained",
    )
    add_heatmap(
        figure,
        axes[0],
        values(["t2m"]),
        ["t2m"],
        "(a) 2 m temperature",
        "RMSE (K)",
    )
    add_heatmap(
        figure,
        axes[1],
        values(["u10", "v10"]),
        ["u10", "v10"],
        "(b) 10 m horizontal wind",
        r"RMSE (m s$^{-1}$)",
    )
    axes[1].set_xlabel("Month in 2020")
    figure.suptitle(
        "Monthly RMSE between surface-station observations and ERA5",
        fontsize=17,
        fontweight="bold",
    )
    return save_figure(
        figure, "figure12_surface_station_monthly_observation_rmse"
    )


def plot_aircraft_monthly() -> tuple[Path, Path]:
    frame = pd.read_csv(ABO_TIMESTEP)
    frame = frame.loc[frame["region"].eq("strict_conus")]
    monthly = frame.groupby(["month", "variable"], as_index=False).agg(
        rmse=("rmse", "mean")
    )

    def values(variables: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [
                    monthly.loc[
                        monthly["variable"].eq(variable)
                        & monthly["month"].eq(month),
                        "rmse",
                    ].iloc[0]
                    for month in range(1, 13)
                ]
                for variable in variables
            ]
        )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.0),
        gridspec_kw={"height_ratios": [2, 4]},
        layout="constrained",
    )
    temperature_variables = ["temperature_500", "temperature_850"]
    wind_variables = [
        "u_component_of_wind_500",
        "u_component_of_wind_850",
        "v_component_of_wind_500",
        "v_component_of_wind_850",
    ]
    add_heatmap(
        figure,
        axes[0],
        values(temperature_variables),
        ["t500", "t850"],
        "(a) Temperature",
        "RMSE (K)",
    )
    add_heatmap(
        figure,
        axes[1],
        values(wind_variables),
        ["u500", "u850", "v500", "v850"],
        "(b) Horizontal wind",
        r"RMSE (m s$^{-1}$)",
    )
    axes[1].set_xlabel("Month in 2020")
    figure.suptitle(
        "Monthly RMSE between aircraft observations and ERA5",
        fontsize=17,
        fontweight="bold",
    )
    return save_figure(
        figure, "figure13_aircraft_monthly_observation_rmse"
    )


def main() -> None:
    configure_style()
    outputs = [plot_surface_station_monthly(), plot_aircraft_monthly()]
    for pdf_path, png_path in outputs:
        print(pdf_path)
        print(png_path)


if __name__ == "__main__":
    main()
