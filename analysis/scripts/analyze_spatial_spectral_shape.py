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

from analysis_paths import required_path

ERA5 = required_path("ERA5_ROOT")
MANIFEST = required_path("EVALUATION_TIMESTEP_MANIFEST")
R_ROOT = required_path("R_ONLY_SAMPLES_ROOT")
RAS_ROOT = required_path("RAS_SAMPLES_ROOT")
OUT_ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = OUT_ROOT / "figures"
OUT_TABLE = OUT_ROOT / "tables"

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
TARGETS = [
    ("2m_temperature", "t2m", "Surface-targeted t2m"),
    ("v_component_of_wind_500", "v500", "Aircraft-targeted v500"),
    ("specific_humidity_850", "q850", "q850 not directly constrained by A or S"),
]
PROTOCOLS = {
    "ERA5 reference": None,
    "R": R_ROOT,
    "R+A+S": RAS_ROOT,
}
COLORS = {"ERA5 reference": "#202124", "R": "#2A6FAD", "R+A+S": "#D97706"}


def sample_path(root: Path, timestep: int) -> Path:
    matches = sorted(root.glob(f"*_t{timestep:04d}_e16_s50.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one sample file for timestep {timestep} in {root}, found {len(matches)}"
        )
    return matches[0]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Nimbus Sans",
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
        }
    )


def load_normalization() -> tuple[np.ndarray, np.ndarray]:
    means = np.load(ERA5 / "normalize_mean.npz")
    stds = np.load(ERA5 / "normalize_std.npz")
    mean = np.asarray([float(np.asarray(means[v]).reshape(-1)[0]) for v in VARIABLES])
    std = np.asarray([float(np.asarray(stds[v]).reshape(-1)[0]) for v in VARIABLES])
    return mean, std


def conus_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.load(ERA5 / "lat.npy").astype(float)
    lon = np.load(ERA5 / "lon.npy").astype(float)
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    iy = np.where((lat >= 24.0) & (lat <= 50.0))[0]
    ix = np.where((lon180 >= -125.0) & (lon180 <= -66.0))[0]
    ix = ix[np.argsort(lon180[ix])]
    return iy, ix, lat[iy], lon180[ix]


def spectrum(field: np.ndarray, bins: np.ndarray, dx_km: float, dy_km: float) -> np.ndarray:
    data = np.asarray(field, dtype=np.float64)
    data = data - np.nanmean(data)
    window = np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1]))
    transformed = np.fft.rfft2(np.nan_to_num(data) * window)
    power = np.abs(transformed) ** 2
    ky = np.fft.fftfreq(data.shape[0], d=dy_km)
    kx = np.fft.rfftfreq(data.shape[1], d=dx_km)
    radial = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    valid = radial > 0
    power = power[valid]
    radial = radial[valid]
    total = power.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(bins) - 1, np.nan)
    binned, _ = np.histogram(radial, bins=bins, weights=power / total)
    return binned


def plot_spectra(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(11.4, 6.1),
        constrained_layout=True,
        sharex="col",
        sharey="row",
    )
    top_labels = ["(a)", "(b)", "(c)"]
    bottom_labels = ["(d)", "(e)", "(f)"]
    for column, (_, short, title) in enumerate(TARGETS):
        subset = table[table["variable"].eq(short)]
        reference = subset[subset["protocol"].eq("ERA5 reference")].set_index(
            "approximate_wavelength_km"
        )["mean_normalized_power_fraction"]
        for protocol in PROTOCOLS:
            protocol_data = subset[subset["protocol"].eq(protocol)].sort_values(
                "approximate_wavelength_km"
            )
            axes[0, column].semilogx(
                protocol_data["approximate_wavelength_km"],
                protocol_data["mean_normalized_power_fraction"],
                color=COLORS[protocol],
                linewidth=2.2 if protocol != "ERA5 reference" else 2.0,
                linestyle="--" if protocol == "ERA5 reference" else "-",
                label=protocol,
            )
            if protocol != "ERA5 reference":
                aligned_reference = reference.loc[
                    protocol_data["approximate_wavelength_km"].to_numpy()
                ].to_numpy()
                ratio = protocol_data["mean_normalized_power_fraction"].to_numpy() / aligned_reference
                axes[1, column].semilogx(
                    protocol_data["approximate_wavelength_km"],
                    ratio,
                    color=COLORS[protocol],
                    linewidth=2.2,
                    label=protocol,
                )
        axes[0, column].set_title(f"{top_labels[column]} {title}")
        axes[1, column].set_title(
            f"{bottom_labels[column]} Ratio to ERA5",
            loc="left",
            fontweight="bold",
            fontsize=10.5,
        )
        axes[1, column].axhline(1.0, color="#555555", linestyle="--", linewidth=1.1)
        axes[1, column].set_xlabel("Approximate wavelength (km)")
        axes[1, column].set_xticks([250, 500, 1000, 2000, 4000])
        axes[1, column].set_xticklabels(["250", "500", "1000", "2000", "4000"])
        axes[1, column].set_ylim(0.0, 1.55)
        for row in range(2):
            axes[row, column].grid(color="#D9DEE3", linewidth=0.7, alpha=0.8)
            axes[row, column].set_axisbelow(True)
    axes[0, 0].set_ylabel("Normalized power fraction")
    axes[1, 0].set_ylabel("Spectral-shape ratio to ERA5")
    axes[0, -1].legend(frameon=True, facecolor="white", edgecolor="#C7CDD3")
    axes[1, -1].legend(frameon=True, facecolor="white", edgecolor="#C7CDD3")
    figure.suptitle(
        "Spatial spectral shape over the CONUS domain at 723 analysis times",
        fontweight="bold",
    )
    figure.savefig(OUT_FIG / "spatial_spectral_shape_conus.pdf")
    figure.savefig(OUT_FIG / "spatial_spectral_shape_conus.png", dpi=300)
    plt.close(figure)


def main() -> None:
    configure_plotting()
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    cached_table = OUT_TABLE / "conus_spatial_spectral_shape_2020.csv"
    if cached_table.exists():
        plot_spectra(pd.read_csv(cached_table))
        print(OUT_FIG / "spatial_spectral_shape_conus.pdf")
        return
    payload = json.loads(MANIFEST.read_text())
    timesteps = [int(t) for t in payload["timesteps"]]
    if len(timesteps) != 723:
        raise ValueError(f"Expected 723 timesteps, found {len(timesteps)}")
    mean, std = load_normalization()
    iy, ix, lat, lon = conus_indices()
    dy_km = 111.0 * float(np.median(np.diff(lat)))
    dx_km = 111.0 * np.cos(np.deg2rad(float(np.mean(lat)))) * float(np.median(np.diff(lon)))
    ky = np.fft.fftfreq(len(iy), d=dy_km)
    kx = np.fft.rfftfreq(len(ix), d=dx_km)
    radial = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    positive = radial[radial > 0]
    bins = np.geomspace(float(positive.min()) * 0.999, float(positive.max()) * 1.001, 10)
    wavelength = 1.0 / np.sqrt(bins[:-1] * bins[1:])

    sums = {(short, protocol): np.zeros(len(bins) - 1) for _, short, _ in TARGETS for protocol in PROTOCOLS}
    counts = {(short, protocol): np.zeros(len(bins) - 1, dtype=int) for _, short, _ in TARGETS for protocol in PROTOCOLS}

    for n, timestep in enumerate(timesteps, start=1):
        if n == 1 or n % 50 == 0 or n == len(timesteps):
            print(f"[{n:03d}/{len(timesteps)}] t{timestep:04d}", flush=True)
        r_path = sample_path(R_ROOT, timestep)
        ras_path = sample_path(RAS_ROOT, timestep)
        r = np.load(r_path, mmap_mode="r")
        ras = np.load(ras_path, mmap_mode="r")
        with h5py.File(ERA5 / "test" / f"2020_{timestep:04d}.h5", "r") as handle:
            for variable, short, _ in TARGETS:
                channel = VARIABLES.index(variable)
                truth = np.asarray(handle["input"][variable], dtype=np.float32)[np.ix_(iy, ix)]
                fields = {
                    "ERA5 reference": truth,
                    "R": np.asarray(r[:, channel][:, iy][:, :, ix], dtype=np.float32).mean(axis=0) * std[channel] + mean[channel],
                    "R+A+S": np.asarray(ras[:, channel][:, iy][:, :, ix], dtype=np.float32).mean(axis=0) * std[channel] + mean[channel],
                }
                for protocol, field in fields.items():
                    values = spectrum(field, bins, dx_km, dy_km)
                    finite = np.isfinite(values)
                    sums[(short, protocol)][finite] += values[finite]
                    counts[(short, protocol)][finite] += 1

    rows = []
    for _, short, _ in TARGETS:
        for protocol in PROTOCOLS:
            values = sums[(short, protocol)] / np.maximum(counts[(short, protocol)], 1)
            for scale, value in zip(wavelength, values):
                rows.append(
                    {
                        "variable": short,
                        "protocol": protocol,
                        "approximate_wavelength_km": scale,
                        "mean_normalized_power_fraction": value,
                        "n_timesteps": int(counts[(short, protocol)].max()),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_TABLE / "conus_spatial_spectral_shape_2020.csv", index=False)
    plot_spectra(table)
    print(OUT_FIG / "spatial_spectral_shape_conus.pdf")


if __name__ == "__main__":
    main()
