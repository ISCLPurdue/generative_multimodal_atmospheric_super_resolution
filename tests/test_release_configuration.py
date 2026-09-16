from __future__ import annotations

import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "reproduction/config/selected_interface_2019.json"
PRIOR_CONFIG_PATH = REPO_ROOT / "reproduction/config/atmospheric_prior_13var.yaml"
EVALUATION_TIMESTEPS = (
    REPO_ROOT / "reproduction/manifests/evaluation_timesteps_2020.json"
)
HOLDOUT_TIMESTEPS = REPO_ROOT / "reproduction/manifests/holdout_timesteps_2020.json"
SUMMARY_ROOT = REPO_ROOT / "analysis/summary_tables"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ReleaseConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.sources = cls.config["aircraft_interface"]["madis_data_source_codes"]
        cls.annual = load_module(
            "annual_wrapper", REPO_ROOT / "reproduction/run_2020_evaluation.py"
        )
        cls.holdout = load_module(
            "holdout_wrapper", REPO_ROOT / "reproduction/run_2020_holdout.py"
        )
        cls.split = load_module(
            "holdout_split",
            REPO_ROOT / "reproduction/create_2020_holdout_split.py",
        )
        cls.targets = load_module(
            "holdout_targets",
            REPO_ROOT / "reproduction/build_holdout_cell_targets.py",
        )

    def production_stub(self):
        return SimpleNamespace(
            EXPERIMENTS={},
            SURFACE_METAR_VARIABLES=[
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
        )

    def test_selected_aircraft_sources_are_explicit(self):
        self.assertEqual(self.sources, [0, 1, 5])
        for configuration in ("R+A", "R+A+S"):
            production = self.production_stub()
            name = self.annual.register_experiment(
                configuration, production, self.sources
            )
            self.assertEqual(
                production.EXPERIMENTS[name]["aircraft_data_sources"], [0, 1, 5]
            )
            self.assertNotIn(
                "aircraft_source_policy_name", production.EXPERIMENTS[name]
            )
            self.assertEqual(
                production.EXPERIMENTS[name]["aircraft_source_filter"], "acars"
            )

    def test_selected_likelihood_parameters_match_manuscript(self):
        self.assertEqual(
            self.config["selected_parameters"],
            {
                "igra_lambda": 1.0,
                "igra_std": 5e-4,
                "igra_gamma": 2e-6,
                "aircraft_lambda": 0.4,
                "aircraft_std": 5e-4,
                "aircraft_gamma": 2e-5,
                "surface_lambda": 0.4,
                "surface_std": 1.25e-4,
                "surface_gamma": 4e-5,
            },
        )

    def test_evaluation_and_holdout_manifests(self):
        evaluation = json.loads(EVALUATION_TIMESTEPS.read_text())["timesteps"]
        holdout = json.loads(HOLDOUT_TIMESTEPS.read_text())["timesteps"]
        self.assertEqual(len(evaluation), 723)
        self.assertEqual(len(set(evaluation)), 723)
        self.assertEqual(len(holdout), 24)
        self.assertEqual(len(set(holdout)), 24)
        self.assertTrue(all(int(value) % 2 == 0 for value in evaluation))
        self.assertTrue(all(int(value) % 2 == 0 for value in holdout))

    def test_holdout_split_applies_conus_before_cell_selection(self):
        locations = np.asarray(
            [[35.0, -100.0], [40.0, -80.0], [51.0, -100.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            self.split.conus_mask(locations), [True, True, False]
        )
        lat = np.linspace(-90.0, 90.0, 128, dtype=np.float32)
        lon = np.linspace(0.0, 360.0, 256, endpoint=False, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aircraft.npz"
            arrays = {}
            for variable in self.split.AIRCRAFT_VARIABLES:
                arrays[f"{variable}_locs"] = locations
                arrays[f"{variable}_vals"] = np.asarray([1.0, 2.0, 3.0])
                arrays[f"{variable}_data_source"] = np.asarray([0, 0, 0])
                arrays[f"{variable}_source_product"] = np.asarray(
                    ["acars", "acars", "acars"]
                )
            np.savez(path, **arrays)
            with np.load(path, allow_pickle=False) as data:
                retained, excluded, rows = self.split.split_aircraft(
                    data,
                    timestep=0,
                    lat_axis=lat,
                    lon_axis=lon,
                    seed=17,
                    holdout_fraction=0.5,
                    allowed_source_codes={0, 1, 5},
                )
        for variable in self.split.AIRCRAFT_VARIABLES:
            kept = retained[f"{variable}_locs"]
            left_out = excluded[f"{variable}_locs"]
            self.assertEqual(kept.shape[0] + left_out.shape[0], 2)
            self.assertTrue(self.split.conus_mask(kept).all())
            self.assertTrue(self.split.conus_mask(left_out).all())
        self.assertTrue(all(row["n_input_obs"] == 3 for row in rows))
        self.assertTrue(all(row["n_eligible_obs"] == 2 for row in rows))

    def test_holdout_target_builder_averages_within_cell(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "madis_aircraft_13var_t0042.npz"
            np.savez(
                path,
                temperature_500_vals=np.asarray([2.0, 4.0, 9.0]),
                temperature_500_era5_flat_cells=np.asarray([7, 7, 11]),
            )
            rows = self.targets.cell_mean_rows(
                path, "aircraft", ["temperature_500"], "CONUS"
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["spatial_domain"], "CONUS")
        self.assertEqual(rows[0]["flat_cell"], 7)
        self.assertEqual(rows[0]["target"], 3.0)
        self.assertEqual(rows[0]["n_obs_in_target"], 2)

    def test_holdout_uses_same_aircraft_sources(self):
        production = self.production_stub()
        name = self.holdout.register_experiment(
            "aircraft", production, self.sources
        )
        self.assertEqual(
            production.EXPERIMENTS[name]["aircraft_data_sources"], [0, 1, 5]
        )
        self.assertEqual(
            production.EXPERIMENTS[name]["obs_space"],
            "aircraft_surface_cell_mean_grid",
        )
        self.assertEqual(
            production.EXPERIMENTS[name]["aircraft_source_filter"], "acars"
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_explicit_grid_paths_override_environment_defaults(self):
        from igra_gen.generating.conditioning_methods import UnifiedOperator

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lat = np.linspace(-90.0, 90.0, 128, dtype=np.float32)
            lon = np.linspace(0.0, 360.0, 256, endpoint=False, dtype=np.float32)
            np.save(root / "lat.npy", lat)
            np.save(root / "lon.npy", lon)
            operator = UnifiedOperator(
                conditioning_type="igra",
                lat_path=root / "lat.npy",
                lon_path=root / "lon.npy",
            )
            np.testing.assert_allclose(operator.igra_op.lat.numpy(), lat)

    def test_satellite_sampling_branch_is_absent(self):
        self.assertFalse(list((REPO_ROOT / "src/igra_gen").glob("sample_lsf_*")))

    def test_aircraft_preprocessing_uses_release_io_module(self):
        scripts = REPO_ROOT / "scripts"
        self.assertTrue((scripts / "madis_aircraft_io.py").exists())
        self.assertFalse((scripts / "diagnose_madis_aircraft_vs_era5.py").exists())
        downloader = (scripts / "download_madis_aircraft_hours.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('download_one "acars"', downloader)
        self.assertNotIn("acarsProfiles", downloader)

    def test_readme_keeps_author_selected_summary_wording(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "evaluates the selected configuration at\n723 analysis times in 2020",
            readme,
        )

    def test_release_citation_uses_exact_manuscript_title_without_doi(self):
        citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn(
            "Generative Atmospheric Super-Resolution across Heterogeneous "
            "Observing Systems through Composable Interfaces",
            citation,
        )
        self.assertNotIn("doi:", citation.lower())
        self.assertIn("version: 1.0.5", citation)
        self.assertIn("date-released: 2026-09-16", citation)
        self.assertIn("license: MIT", citation)
        license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("MIT License\n"))
        self.assertIn("Copyright (c) 2026 Yang Xu and contributors", license_text)

    def test_internal_release_documents_are_not_packaged(self):
        for name in (
            "PAPER_ALIGNMENT_AUDIT.md",
            "RELEASE_CHECKLIST.md",
            "RELEASE_NOTES_DRAFT.md",
        ):
            self.assertFalse((REPO_ROOT / name).exists())

    def test_reported_annual_and_probabilistic_summaries(self):
        annual = read_csv(SUMMARY_ROOT / "annual_rmse_14day_intervals_by_region.csv")
        conus_all = next(
            row
            for row in annual
            if row["region"] == "strict_conus" and row["variable_set"] == "all13"
        )
        self.assertEqual(int(conus_all["n_evaluation_times"]), 723)
        self.assertEqual(int(conus_all["block_days"]), 14)
        self.assertAlmostEqual(float(conus_all["mean_effect_pct"]), -9.2363, places=3)
        self.assertLess(float(conus_all["ci95_primary_high_pct"]), 0.0)
        self.assertFalse(any("paired_t" in column for column in conus_all))

        per_variable = read_csv(
            SUMMARY_ROOT / "annual_rmse_14day_intervals_by_variable.csv"
        )
        conus_variables = [
            row for row in per_variable if row["region"] == "strict_conus"
        ]
        self.assertEqual(len(conus_variables), 13)
        self.assertTrue(
            all(float(row["ci95_primary_high_pct"]) < 0.0 for row in conus_variables)
        )

        crps = read_csv(SUMMARY_ROOT / "crps_14day_intervals.csv")
        conus_crps = next(
            row
            for row in crps
            if row["region"] == "strict_conus"
            and row["evaluation_group"] == "all_13_variables"
        )
        self.assertAlmostEqual(float(conus_crps["mean_crps_pct_change"]), -9.9762, places=3)

    def test_reported_heldout_summaries(self):
        rows = read_csv(SUMMARY_ROOT / "heldout_family_summary.csv")
        self.assertTrue(rows)
        self.assertEqual({row["method"] for row in rows}, {"heldout80"})
        selected = {
            row["family"]: row
            for row in rows
            if row["method"] == "heldout80"
        }
        self.assertAlmostEqual(float(selected["surface"]["mean_effect_pct"]), -13.4992, places=3)
        self.assertAlmostEqual(float(selected["aircraft"]["mean_effect_pct"]), -11.7137, places=3)
        self.assertEqual(int(selected["aircraft"]["n_timesteps"]), 24)
        self.assertAlmostEqual(
            float(selected["aircraft"]["improved_timestep_fraction"]),
            23 / 24,
        )
        self.assertTrue(
            all(float(row["ci95_high_pct"]) < 0.0 for row in selected.values())
        )
        manuscript_table = (
            REPO_ROOT / "analysis/manuscript_tables/heldout_observation_evaluation.tex"
        ).read_text(encoding="utf-8")
        self.assertIn("2020 CONUS observations", manuscript_table)
        self.assertIn("$-11.71\\%$ & $[-14.48,-9.00]\\%$ & $23/24$", manuscript_table)

    def test_resolved_prior_configuration_is_packaged(self):
        text = PRIOR_CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("_target_: igra_gen.models.songunet.SongUNet", text)
        self.assertIn("model_channels: 64", text)
        self.assertIn("2m_temperature", text)
        self.assertIn("specific_humidity_850", text)

    def test_training_pipeline_is_not_packaged(self):
        self.assertFalse((REPO_ROOT / "src/igra_gen/train.py").exists())
        self.assertFalse((REPO_ROOT / "src/igra_gen/training").exists())
        self.assertFalse((REPO_ROOT / "src/igra_gen/configs/train.yaml").exists())

    def test_composition_summary_schema_and_reported_values(self):
        rows = read_csv(SUMMARY_ROOT / "composition_summary.csv")
        self.assertEqual(len(rows), 9)
        self.assertEqual(
            set(rows[0]),
            {
                "region",
                "configuration",
                "n_analysis_times",
                "all13_mean_rmse_change_pct",
                "surface3_mean_rmse_change_pct",
                "aircraft6_mean_rmse_change_pct",
            },
        )
        conus = {
            row["configuration"]: row
            for row in rows
            if row["region"] == "strict_conus"
        }
        self.assertEqual(set(conus), {"R+A", "R+S", "R+A+S"})
        self.assertTrue(all(int(row["n_analysis_times"]) == 723 for row in conus.values()))
        self.assertAlmostEqual(
            float(conus["R+A"]["all13_mean_rmse_change_pct"]), -4.4633, places=3
        )
        self.assertAlmostEqual(
            float(conus["R+S"]["all13_mean_rmse_change_pct"]), -5.3493, places=3
        )
        self.assertAlmostEqual(
            float(conus["R+A+S"]["all13_mean_rmse_change_pct"]), -9.2363, places=3
        )

    def test_removed_ttest_table_is_not_packaged(self):
        self.assertFalse(
            list((REPO_ROOT / "analysis/manuscript_tables").glob("*paired_ttest*"))
        )


if __name__ == "__main__":
    unittest.main()
