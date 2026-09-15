#!/usr/bin/env python3
"""Plot polished common-bin annual R/A/S observation distributions.

The layout retains the comfortable overall proportions of the v111 preview,
while the typography, map spacing, CONUS annotation, and shared colorbar are
drawn together so they remain visually consistent at manuscript size.
"""

from __future__ import annotations

import csv
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/home/xu2279/.tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from replot_observation_geometry_annual_v111 import (
    ABO_COUNTS,
    CONUS,
    IGRA_PKL,
    METAR_INVENTORY,
    SHP,
    draw_basemap,
    draw_conus_domain,
    new_figure,
    read_polygon_shp,
    style_axes,
    style_colorbar,
    style_legend,
)


VERSION_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = VERSION_ROOT / "figures"
TABLE_DIR = VERSION_ROOT / "tables"
BIN_WIDTH_DEG = 0.5
MAP_YLIM = (-60.0, 80.0)
COMMON_COLOR_LABEL = "log10 annual observation count per 0.5° bin"


def bin_centers(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=np.float64)
    lon = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat_bin = np.floor(lat / BIN_WIDTH_DEG) * BIN_WIDTH_DEG + BIN_WIDTH_DEG / 2
    lon_bin = np.floor(lon / BIN_WIDTH_DEG) * BIN_WIDTH_DEG + BIN_WIDTH_DEG / 2
    return lat_bin, lon_bin


def restrict_to_map(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        frame["lat"].between(*MAP_YLIM)
        & frame["lon"].between(-180.0, 180.0)
        & (frame["count"] > 0)
    ].copy()


def load_radiosonde_bins() -> pd.DataFrame:
    with IGRA_PKL.open("rb") as handle:
        data = pickle.load(handle)
    rows: list[tuple[float, float]] = []
    for timestep in range(0, len(data), 2):
        blocks = [
            np.asarray(channel, dtype=np.float64)
            for channel in data[timestep][0][0]
            if len(channel)
        ]
        if not blocks:
            continue
        # Count one available profile per station and analysis time after
        # pooling variables, then accumulate those profile occurrences by bin.
        locations = np.unique(np.concatenate(blocks, axis=0), axis=0)
        lat_bin, lon_bin = bin_centers(locations[:, 0], locations[:, 1])
        rows.extend(zip(lat_bin, lon_bin))
    frame = pd.DataFrame(rows, columns=("lat", "lon"))
    frame = frame.value_counts().rename("count").reset_index()
    return restrict_to_map(frame)


def load_aircraft_bins() -> pd.DataFrame:
    frame = pd.read_csv(ABO_COUNTS).rename(columns={"n_obs_records": "count"})
    return restrict_to_map(frame[["lat", "lon", "count"]])


def load_surface_bins() -> pd.DataFrame:
    stations = pd.read_csv(METAR_INVENTORY)
    lat_bin, lon_bin = bin_centers(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    frame = pd.DataFrame(
        {"lat": lat_bin, "lon": lon_bin, "count": stations["n_obs_records"].to_numpy()}
    )
    frame = frame.groupby(["lat", "lon"], as_index=False)["count"].sum()
    return restrict_to_map(frame)


def save_panel(
    segments: list[np.ndarray],
    frame: pd.DataFrame,
    title: str,
    marker_size: float,
    filename: str,
    norm: colors.Normalize,
) -> Path:
    fig, ax, cax = new_figure(with_colorbar=True)
    draw_basemap(ax, segments)
    scatter = ax.scatter(
        frame["lon"],
        frame["lat"],
        s=marker_size,
        c=np.log10(frame["count"].to_numpy()),
        cmap="viridis",
        norm=norm,
        alpha=0.78,
        linewidths=0,
        label=f"Occupied bins, N={len(frame)}",
        zorder=3,
    )
    draw_conus_domain(ax)
    style_axes(ax, title)
    style_legend(ax)
    colorbar = fig.colorbar(scatter, cax=cax)
    style_colorbar(colorbar, COMMON_COLOR_LABEL)
    output = FIG_DIR / filename
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)
    return output


def save_combined_preview(panel_paths: list[Path]) -> Path:
    labels = ("(a) Radiosonde (R)", "(b) Aircraft (A)", "(c) Surface station (S)")
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 3.55), facecolor="white")
    for ax, path, label in zip(axes, panel_paths, labels):
        ax.imshow(plt.imread(path))
        ax.set_axis_off()
        ax.set_title(label, fontsize=16, pad=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01, wspace=0.015)
    output = FIG_DIR / "figure2_observation_geometry_common_bins_preview.png"
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)
    return output


def save_shared_colorbar_figure(
    segments: list[np.ndarray],
    frames: dict[str, pd.DataFrame],
    norm: colors.Normalize,
) -> Path:
    titles = {
        "R": "(a) Radiosonde (R)",
        "A": "(b) Aircraft (A)",
        "S": "(c) Surface station (S)",
    }
    marker_sizes = {"R": 12.0, "A": 1.7, "S": 5.5}

    # The v111 preview used an 18 x 3.55 canvas.  Fixed axes positions retain
    # that proportion without allowing the shared colorbar to compress the
    # third map or create uneven panel-to-panel spacing.
    fig = plt.figure(figsize=(18.0, 3.55), facecolor="white")
    axes = [
        fig.add_axes([0.045, 0.175, 0.278, 0.655]),
        fig.add_axes([0.350, 0.175, 0.278, 0.655]),
        fig.add_axes([0.655, 0.175, 0.278, 0.655]),
    ]
    cax = fig.add_axes([0.949, 0.175, 0.011, 0.655])
    last_scatter = None
    for ax, source in zip(axes, ("R", "A", "S")):
        frame = frames[source]
        draw_basemap(ax, segments)
        last_scatter = ax.scatter(
            frame["lon"],
            frame["lat"],
            s=marker_sizes[source],
            c=np.log10(frame["count"].to_numpy()),
            cmap="viridis",
            norm=norm,
            alpha=0.78,
            linewidths=0,
            zorder=3,
        )
        # A restrained domain annotation is clearer than a legend whose
        # rectangle handle visually competes with the data.
        lat0, lat1, lon0, lon1 = CONUS
        ax.add_patch(
            patches.Rectangle(
                (lon0, lat0),
                lon1 - lon0,
                lat1 - lat0,
                fill=False,
                linestyle=(0, (5, 3)),
                linewidth=1.15,
                edgecolor="#303030",
                zorder=4,
            )
        )
        ax.set_title(
            titles[source],
            fontsize=16,
            fontweight="normal",
            fontfamily="Nimbus Sans",
            pad=9,
        )
        ax.set_xlabel("Longitude", fontsize=12, labelpad=4)
        ax.tick_params(axis="both", labelsize=10.5, width=0.8, length=3.5)
        if source == "R":
            ax.set_ylabel("Latitude", fontsize=12, labelpad=4)
        else:
            ax.set_ylabel("")
    assert last_scatter is not None
    colorbar = fig.colorbar(last_scatter, cax=cax)
    colorbar.set_label(
        COMMON_COLOR_LABEL,
        fontsize=11.5,
        labelpad=9,
        fontfamily="Nimbus Sans",
    )
    colorbar.ax.tick_params(labelsize=10, width=0.8, length=3.5)
    colorbar.outline.set_linewidth(0.7)
    output = FIG_DIR / "figure2_observation_geometry_annual_polished.png"
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    segments = read_polygon_shp(SHP)
    frames = {
        "R": load_radiosonde_bins(),
        "A": load_aircraft_bins(),
        "S": load_surface_bins(),
    }
    common_max = max(float(np.log10(frame["count"]).max()) for frame in frames.values())
    norm = colors.Normalize(vmin=0.0, vmax=common_max)
    outputs = [
        save_panel(segments, frames["R"], "Radiosonde observations (IGRA)\n2020, 00/12 UTC", 24, "map_radiosonde_global_ras.png", norm),
        save_panel(segments, frames["A"], "Aircraft observations (MADIS ABO)\n2020, 00/12 UTC", 3.5, "map_aircraft_global_ras.png", norm),
        save_panel(segments, frames["S"], "Surface-station observations (MADIS METAR)\n2020, 00/12 UTC", 12, "map_surface_station_global_ras.png", norm),
    ]
    preview = save_combined_preview(outputs)
    shared_colorbar_figure = save_shared_colorbar_figure(segments, frames, norm)

    summary = TABLE_DIR / "figure2_common_bin_counts.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "bins", "annual_observation_count", "maximum_bin_count"))
        writer.writeheader()
        for source, frame in frames.items():
            writer.writerow(
                {
                    "source": source,
                    "bins": len(frame),
                    "annual_observation_count": int(frame["count"].sum()),
                    "maximum_bin_count": int(frame["count"].max()),
                }
            )
    for path in (*outputs, preview, shared_colorbar_figure, summary):
        print(path)


if __name__ == "__main__":
    main()
