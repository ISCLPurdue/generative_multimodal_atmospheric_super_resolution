#!/usr/bin/env python3
"""Draw all-13-variable runtime-guidance footprints with a white display floor."""

from __future__ import annotations

import os
from pathlib import Path
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np

from analysis_paths import required_path

TRACE = required_path("RUNTIME_GUIDANCE_TRACE_NPZ")
SHAPEFILE = required_path("NATURAL_EARTH_SHAPEFILE")
FIGURES = Path(__file__).resolve().parents[1] / "figures"

STEP_INDICES = [0, 25, 49]
STEP_LABELS = ["First reverse step", "Middle reverse step", "Final reverse step"]
SOURCES = ["R", "A", "S"]
ARRAY_KEYS = {"R": "r", "A": "a", "S": "s"}
SOURCE_LABELS = {
    "R": "Radiosonde (R)",
    "A": "Aircraft (A)",
    "S": "Surface station (S)",
}
SOURCE_COLORS = {"R": "#2A6FAD", "A": "#D97706", "S": "#0F8F83"}

VIEW = (-132.0, -58.0, 18.0, 56.0)
CONUS_BOX = (24.0, 50.0, -125.0, -66.0)
DISPLAY_FLOOR = 3.0e-3
HIGH_PERCENTILE = 99.9

CMAP = LinearSegmentedColormap.from_list(
    "guidance_yellow_green",
    ["#ffffff", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679", "#31a354", "#006837"],
)
CMAP.set_bad("#ffffff")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def read_polygon_shp(path: Path) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    with path.open("rb") as handle:
        handle.seek(100)
        while True:
            header = handle.read(8)
            if len(header) < 8:
                break
            _, content_len_words = struct.unpack(">2i", header)
            content = handle.read(content_len_words * 2)
            if len(content) < 44:
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type not in (5, 15, 25):
                continue
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_offset = 44
            points_offset = parts_offset + 4 * num_parts
            if len(content) < points_offset + 16 * num_points:
                continue
            parts = list(
                struct.unpack(f"<{num_parts}i", content[parts_offset:points_offset])
            )
            parts.append(num_points)
            points = np.frombuffer(
                content[points_offset : points_offset + 16 * num_points],
                dtype="<f8",
            ).reshape(-1, 2)
            for start, stop in zip(parts[:-1], parts[1:]):
                segment = points[start:stop]
                if len(segment) >= 2:
                    segments.append(segment.copy())
    return segments


def segment_in_view(segment: np.ndarray) -> bool:
    lon_min, lon_max, lat_min, lat_max = VIEW
    return not (
        segment[:, 0].max() < lon_min
        or segment[:, 0].min() > lon_max
        or segment[:, 1].max() < lat_min
        or segment[:, 1].min() > lat_max
    )


def draw_boundaries(axis: plt.Axes, segments: list[np.ndarray]) -> None:
    for segment in segments:
        if segment_in_view(segment):
            axis.plot(
                segment[:, 0],
                segment[:, 1],
                color="#43545d",
                linewidth=0.42,
                alpha=0.82,
                zorder=4,
            )
    lat0, lat1, lon0, lon1 = CONUS_BOX
    axis.plot(
        [lon0, lon1, lon1, lon0, lon0],
        [lat0, lat0, lat1, lat1, lat0],
        color="#202830",
        linestyle="--",
        linewidth=1.05,
        zorder=5,
    )


def shared_log_limits(data: np.lib.npyio.NpzFile) -> tuple[float, float]:
    pooled = np.concatenate(
        [
            np.asarray(data[ARRAY_KEYS[source]][step_index], dtype=float).ravel()
            for step_index in STEP_INDICES
            for source in SOURCES
        ]
    )
    positive = pooled[np.isfinite(pooled) & (pooled > 0)]
    return DISPLAY_FLOOR, float(np.percentile(positive, HIGH_PERCENTILE))


def main() -> None:
    configure_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = np.load(TRACE)
    steps = data["steps"]
    sigmas = data["sigmas"]
    lon = data["lon_signed"]
    lat = data["lat"]
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    segments = read_polygon_shp(SHAPEFILE)
    shared_vmin, shared_vmax = shared_log_limits(data)

    figure, axes = plt.subplots(
        3,
        3,
        figsize=(10.8, 7.35),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    image = None
    panel_index = 0
    for row, step_index in enumerate(STEP_INDICES):
        for col, source in enumerate(SOURCES):
            axis = axes[row, col]
            raw_guidance = np.asarray(
                data[ARRAY_KEYS[source]][step_index], dtype=float
            )
            displayed_guidance = np.ma.masked_less_equal(
                raw_guidance, DISPLAY_FLOOR
            )
            image = axis.imshow(
                np.minimum(displayed_guidance, shared_vmax),
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap=CMAP,
                norm=LogNorm(vmin=shared_vmin, vmax=shared_vmax),
                interpolation="nearest",
                zorder=1,
            )
            draw_boundaries(axis, segments)
            axis.set_xlim(VIEW[0], VIEW[1])
            axis.set_ylim(VIEW[2], VIEW[3])
            axis.grid(color="#7f8c8d", linewidth=0.35, alpha=0.20, zorder=0)
            axis.text(
                0.018,
                0.965,
                f"({chr(ord('a') + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="#172026",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
                zorder=6,
            )
            panel_index += 1

            if row == 0:
                axis.set_title(
                    SOURCE_LABELS[source],
                    color=SOURCE_COLORS[source],
                    fontweight="bold",
                    pad=7,
                )
            if col == 0:
                sigma_text = f"{float(sigmas[step_index]):.3g}"
                axis.set_ylabel(
                    f"{STEP_LABELS[row]}\n"
                    f"step {int(steps[step_index])}, $\\sigma={sigma_text}$\n"
                    "Latitude",
                    labelpad=8,
                )
    figure.subplots_adjust(
        left=0.105,
        right=0.985,
        top=0.945,
        bottom=0.185,
        wspace=0.055,
        hspace=0.10,
    )
    figure.text(0.545, 0.137, "Longitude", ha="center", va="center")
    colorbar_axis = figure.add_axes([0.275, 0.052, 0.55, 0.024])
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        "Raw-guidance norm across all 13 variables "
        r"($\leq 3\times10^{-3}$ shown white; shared logarithmic scale)"
    )
    colorbar.set_ticks([3e-3, 1e-2, 1.0, 1e2, 1e3])
    colorbar.set_ticklabels(
        [r"$3\times10^{-3}$", r"$10^{-2}$", r"$10^{0}$", r"$10^{2}$", r"$10^{3}$"]
    )

    stem = FIGURES / "runtime_guidance_spatial_footprints"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
