#!/usr/bin/env python3
"""Plot annual R/A/S observation distributions using common spatial bins."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile


VERSION_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = VERSION_ROOT / "figures"
TABLE_DIR = VERSION_ROOT / "tables"
CONUS = (24.0, 50.0, -125.0, -66.0)
BIN_WIDTH_DEG = 0.5
MAP_YLIM = (-60.0, 80.0)
COMMON_COLOR_LABEL = "log10 annual observation count per 0.5° bin"


def read_polygon_shp(path: Path) -> list[np.ndarray]:
    """Return polygon segments from a Natural Earth-style shapefile."""
    segments: list[np.ndarray] = []
    with shapefile.Reader(str(path)) as reader:
        for shape in reader.shapes():
            points = np.asarray(shape.points, dtype=np.float64)
            boundaries = list(shape.parts) + [len(points)]
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                if stop - start >= 2:
                    segments.append(points[start:stop])
    return segments


def draw_basemap(ax: plt.Axes, segments: list[np.ndarray]) -> None:
    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], color="#707070", linewidth=0.45, zorder=1)


def draw_conus_domain(ax: plt.Axes) -> None:
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
            label="CONUS domain",
            zorder=4,
        )
    )


def new_figure(*, with_colorbar: bool) -> tuple[plt.Figure, plt.Axes, plt.Axes | None]:
    fig = plt.figure(figsize=(7.0, 3.55), facecolor="white")
    ax = fig.add_axes([0.08, 0.16, 0.78 if with_colorbar else 0.88, 0.72])
    cax = fig.add_axes([0.89, 0.16, 0.025, 0.72]) if with_colorbar else None
    return fig, ax, cax


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(*MAP_YLIM)
    ax.set_xlabel("Longitude", fontfamily="Nimbus Sans")
    ax.set_ylabel("Latitude", fontfamily="Nimbus Sans")
    ax.set_title(title, fontsize=14, fontfamily="Nimbus Sans", pad=8)
    ax.tick_params(axis="both", labelsize=9, width=0.8, length=3.5)


def style_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(label, fontsize=10.5, labelpad=8, fontfamily="Nimbus Sans")
    colorbar.ax.tick_params(labelsize=9, width=0.8, length=3.5)
    colorbar.outline.set_linewidth(0.7)


def style_legend(ax: plt.Axes) -> None:
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9)


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


def load_radiosonde_bins(igra_pkl: Path) -> pd.DataFrame:
    with igra_pkl.open("rb") as handle:
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


def load_aircraft_bins(aircraft_counts: Path) -> pd.DataFrame:
    frame = pd.read_csv(aircraft_counts).rename(columns={"n_obs_records": "count"})
    return restrict_to_map(frame[["lat", "lon", "count"]])


def load_surface_bins(surface_inventory: Path) -> pd.DataFrame:
    stations = pd.read_csv(surface_inventory)
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
    assert cax is not None
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

    # Fixed axes positions prevent the shared colorbar from compressing the
    # third map or creating uneven panel-to-panel spacing.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--igra-pkl", type=Path, required=True)
    parser.add_argument("--aircraft-counts", type=Path, required=True)
    parser.add_argument("--surface-inventory", type=Path, required=True)
    parser.add_argument("--shapefile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=VERSION_ROOT)
    args = parser.parse_args()

    global FIG_DIR, TABLE_DIR
    FIG_DIR = args.output_dir / "figures"
    TABLE_DIR = args.output_dir / "tables"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    segments = read_polygon_shp(args.shapefile)
    frames = {
        "R": load_radiosonde_bins(args.igra_pkl),
        "A": load_aircraft_bins(args.aircraft_counts),
        "S": load_surface_bins(args.surface_inventory),
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
