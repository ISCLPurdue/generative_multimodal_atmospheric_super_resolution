# Analysis artifacts

This directory connects the manuscript's reported evidence to the analysis
code and compact outputs used to produce it. It is an analysis snapshot, not a
redistribution of the ERA5 fields, NOAA observations, diffusion checkpoint, or
posterior sample arrays described in `../DATA.md`.

## Contents

- `scripts/`: final annual, held-out-observation, probabilistic, dispersion,
  composition, spatial, and plotting analyses.
- `summary_tables/`: compact CSV outputs underlying the principal numerical
  claims in the manuscript.
- `manuscript_tables/`: the LaTeX tables included by the manuscript.

## Claim-to-artifact map

| Manuscript evidence | Analysis script | Compact output |
|---|---|---|
| Annual R+A+S RMSE changes, per-variable effects, and 3-, 7-, and 14-day moving-block sensitivity | `analyze_annual_rmse.py`, `bootstrap_annual_rmse.py` | `annual_rmse_14day_intervals_by_region.csv`, `annual_rmse_14day_intervals_by_variable.csv` |
| R/A/S composition and complementarity | `analyze_composition.py` | `composition_summary.csv`, `complementarity_summary.csv` |
| Errors at observations excluded from conditioning | `analyze_heldout_observations.py` | `heldout_family_summary.csv`, `heldout_variable_summary.csv` |
| CRPS and fair-CRPS analyses | `analyze_probabilistic_metrics.py`, `bootstrap_crps.py` | `probabilistic_group_summary.csv`, `crps_14day_intervals.csv` |
| Spread-skill ratios and rank histograms | `analyze_spread_skill_and_rank.py` | `spread_skill_by_group.csv`, `spread_skill_by_variable.csv`, `rank_histogram_by_group.csv` |
| Spatial, spectral, observation-geometry, field-example, and guidance diagnostics | the corresponding `plot_*` or `analyze_spatial_*` scripts | final manuscript figures are generated from external analysis inputs |

## Rerun boundary

The numerical summaries are small enough to distribute directly. Recomputing
them from scratch requires the external ERA5 files, observation products,
normalization files, checkpoint, and posterior samples. Analysis inputs are
supplied through the environment variables named in each script. Numerical
analyses use subdirectories of `analysis/generated/` by default; standalone
plotting scripts write to `analysis/figures/` or `analysis/tables/` unless an
output variable is provided.

For example, the annual R+A+S analysis expects directories containing the
ERA5 fields, R-only samples, and R+A+S samples, together with the evaluation
manifest:

```bash
ERA5_ROOT=/path/to/era5 \
R_ONLY_SAMPLES_ROOT=/path/to/r_only/samples \
RAS_SAMPLES_ROOT=/path/to/ras/samples \
EVALUATION_TIMESTEP_MANIFEST=reproduction/manifests/evaluation_timesteps_2020.json \
python analysis/scripts/analyze_annual_rmse.py
```

Each standalone script names any additional required environment variable in
its startup error. Scripts with a command-line interface, such as
`analyze_composition.py` and `plot_observation_geometry.py`, list their inputs
under `--help`.

The statistical inference follows the manuscript: paired analysis-time RMSE
or CRPS changes are summarized with moving-block bootstrap intervals.
