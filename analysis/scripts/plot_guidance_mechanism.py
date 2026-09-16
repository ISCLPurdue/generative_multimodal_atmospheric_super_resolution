#!/usr/bin/env python3
"""Build the static and runtime R/A/S mechanism diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from analysis_paths import required_path

STATIC = required_path("GUIDANCE_STATIC_TABLE_ROOT")
TRACE = required_path("GUIDANCE_TRACE_TABLE_ROOT")
OUT = Path(__file__).resolve().parents[1] / "figures"

FACTORS = ("R", "A", "S")
FACTOR_COLORS = {"R": "#2A6FAD", "A": "#D97706", "S": "#0F8F83"}
PAIR_COLORS = {"R vs A": "#A23E16", "R vs S": "#087F73", "A vs S": "#5744B5"}
GROUPS = {
    "Surface-targeted (t2m/u10/v10)": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ],
    "Geopotential height (z500/z850)": ["geopotential_500", "geopotential_850"],
    "Aircraft-targeted (t/u/v at 500/850 hPa)": [
        "u_component_of_wind_500",
        "u_component_of_wind_850",
        "v_component_of_wind_500",
        "v_component_of_wind_850",
        "temperature_500",
        "temperature_850",
    ],
    "Specific humidity (q500/q850)": ["specific_humidity_500", "specific_humidity_850"],
}
GROUP_COLORS = {
    "Surface-targeted (t2m/u10/v10)": "#3B78A8",
    "Geopotential height (z500/z850)": "#9CC7DE",
    "Aircraft-targeted (t/u/v at 500/850 hPa)": "#E58A20",
    "Specific humidity (q500/q850)": "#59A14F",
}


def configure() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / f"{stem}.pdf")


def factor_tick_labels(axis: plt.Axes) -> None:
    axis.set_yticks(np.arange(3), FACTORS)
    for label, factor in zip(axis.get_yticklabels(), FACTORS):
        label.set_color(FACTOR_COLORS[factor])
        label.set_fontweight("bold")
        label.set_fontsize(12)


def static_group_shares() -> dict[str, np.ndarray]:
    data = np.load(STATIC / "static_likelihood_scores_t0000.npz", allow_pickle=True)
    names = [str(value) for value in data["channel_names"]]
    score_keys = {"R": "r_score", "A": "a_score", "S": "s_score"}
    shares: dict[str, np.ndarray] = {}
    for factor, key in score_keys.items():
        score = np.asarray(data[key], dtype=np.float64)
        energy = []
        for variables in GROUPS.values():
            idx = [names.index(variable) for variable in variables]
            energy.append(float(np.sum(score[idx] ** 2)))
        energy_array = np.asarray(energy)
        shares[factor] = 100.0 * energy_array / max(energy_array.sum(), np.finfo(float).tiny)
    return shares


def static_cosines() -> np.ndarray:
    matrix = np.full((3, 3), np.nan)
    pair_order = ["R_vs_A", "R_vs_S", "A_vs_S"]
    group_order = ["all13", "surface3", "aircraft6"]
    for row in rows(STATIC / "gradient_pairwise_cosines.csv"):
        if row["region"] != "strict_conus":
            continue
        if row["pair"] in pair_order and row["group"] in group_order:
            matrix[pair_order.index(row["pair"]), group_order.index(row["group"])] = float(row["cosine"])
    return matrix


def plot_direct_mechanism() -> None:
    shares = static_group_shares()
    cosines = static_cosines()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.05),
        gridspec_kw={"width_ratios": [1.2, 0.95]},
    )

    # (a) Variable support.
    ax = axes[0]
    left = np.zeros(3)
    for group_index, (group, color) in enumerate(GROUP_COLORS.items()):
        values = np.asarray([shares[factor][group_index] for factor in FACTORS])
        bars = ax.barh(np.arange(3), values, left=left, color=color, height=0.56, label=group)
        for row_index, (bar, value) in enumerate(zip(bars, values)):
            if value >= 10:
                ax.text(
                    left[row_index] + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if group != "Geopotential height (z500/z850)" else "#1F2937",
                    fontsize=9.5,
                    fontweight="bold",
                )
        left += values
    factor_tick_labels(ax)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of squared clean-state gradient norm (%)")
    ax.set_title("(a) Gradient-norm allocation by variable group", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=8.7,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    # (b) Directional overlap.
    ax = axes[1]
    finite = np.abs(cosines[np.isfinite(cosines)])
    bound = max(0.25, float(finite.max()) * 1.08)
    image = ax.imshow(
        cosines,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
    )
    ax.set_xticks(np.arange(3), ["All 13", "Surface-\ntargeted", "Aircraft-\ntargeted"])
    ax.set_yticks(np.arange(3), ["R vs A", "R vs S", "A vs S"])
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(3):
            value = cosines[i, j]
            text = "N/A" if not np.isfinite(value) else f"{value:+.2f}"
            color = "white" if np.isfinite(value) and abs(value) > 0.13 else "#172033"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    ax.set_title("(b) Directional overlap over CONUS", loc="left", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Cosine similarity", fontsize=10)

    fig.suptitle(
        "Source likelihood gradients differ before denoiser pullback",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=2.0)
    save(fig, "source_likelihood_gradient_geometry")


def runtime_data() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    step_rows = rows(TRACE / "runtime_step_score_metrics_t0000.csv")
    steps = np.asarray([int(row["step"]) for row in step_rows])
    sigmas = np.asarray([float(row["sigma"]) for row in step_rows])
    cosines = {
        "R vs A": np.asarray([float(row["cos_R_vs_A_all13_conus"]) for row in step_rows]),
        "R vs S": np.asarray([float(row["cos_R_vs_S_all13_conus"]) for row in step_rows]),
        "A vs S": np.asarray([float(row["cos_A_vs_S_all13_conus"]) for row in step_rows]),
    }
    norms = {
        factor: np.asarray([float(row[f"{factor}_raw_l2"]) for row in step_rows])
        for factor in FACTORS
    }
    total = sum(norms.values())
    strength = {factor: 100.0 * norms[factor] / np.maximum(total, np.finfo(float).tiny) for factor in FACTORS}
    return steps, sigmas, cosines, strength


def add_sigma_axis(axis: plt.Axes, steps: np.ndarray, sigmas: np.ndarray) -> None:
    selected = np.asarray([0, 10, 25, 40, 49])
    top = axis.secondary_xaxis("top")
    top.set_xticks(steps[selected], [f"{sigmas[i]:.3g}" for i in selected])
    top.set_xlabel(r"Noise level $\sigma$", labelpad=4)


def endpoint_legend(
    axis: plt.Axes,
    *,
    loc: str,
    title: str,
    ncol: int = 1,
) -> None:
    legend = axis.legend(
        loc=loc,
        ncol=ncol,
        title=title,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#BCC3CC",
        borderpad=0.55,
        labelspacing=0.35,
        handlelength=1.8,
        handletextpad=0.55,
        columnspacing=0.9,
    )
    legend.get_title().set_fontweight("bold")


def plot_runtime_mechanism() -> None:
    steps, sigmas, cosines, strength = runtime_data()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.55))

    def cosine_label(pair: str, value: float) -> str:
        if abs(value) < 1.0e-3:
            return f"{pair}  {value:+.2e}"
        return f"{pair}  {value:+.4f}"

    ax = axes[0]
    for pair, values in cosines.items():
        ax.plot(
            steps,
            values,
            color=PAIR_COLORS[pair],
            lw=2.25,
            label=cosine_label(pair, values[-1]),
        )
        ax.scatter(
            [steps[-1]],
            [values[-1]],
            color=PAIR_COLORS[pair],
            s=27,
            zorder=5,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.axhline(0.0, color="#5B6470", lw=1.0, ls="--")
    ax.set_ylim(-0.42, 1.0)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("CONUS-domain cosine similarity")
    ax.set_title("(a) Directional alignment after pullback", loc="left", fontweight="bold")
    ax.grid(alpha=0.2)
    endpoint_legend(ax, loc="lower left", title="Final-step cosine")
    add_sigma_axis(ax, steps, sigmas)

    ax = axes[1]
    ax.stackplot(
        steps,
        [strength[factor] for factor in FACTORS],
        colors=[FACTOR_COLORS[factor] for factor in FACTORS],
        labels=[f"{factor}  {strength[factor][-1]:.2f}%" for factor in FACTORS],
        alpha=0.93,
    )
    ax.set_ylim(0, 100)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Share of summed raw-guidance norms (%)")
    ax.set_title("(b) Relative raw-guidance magnitude", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    endpoint_legend(ax, loc="lower left", title="Final-step fraction")
    add_sigma_axis(ax, steps, sigmas)

    fig.suptitle(
        "Denoiser pullback changes source-gradient alignment and magnitude",
        fontsize=15,
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout(w_pad=2.0)
    save(fig, "runtime_source_guidance_coupling")


def main() -> None:
    configure()
    plot_direct_mechanism()
    plot_runtime_mechanism()


if __name__ == "__main__":
    main()
