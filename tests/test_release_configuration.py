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
            self.assertNotIn(
                "superob", production.EXPERIMENTS[name]["obs_space"]
            )
            self.assertEqual(
                production.EXPERIMENTS[name]["aircraft_source_filter"], "acars"
            )

    def test_frozen_likelihood_parameters_match_manuscript(self):
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

    def test_dormant_satellite_branch_is_absent(self):
        self.assertFalse(
            (REPO_ROOT / "src/igra_gen/sample_lsf_airtemp_common_native.py").exists()
        )

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
        self.assertEqual(int(conus_all["n_full723"]), 723)
        self.assertEqual(int(conus_all["block_days"]), 14)
        self.assertAlmostEqual(float(conus_all["mean_effect_pct_full723"]), -9.2363, places=3)
        self.assertLess(float(conus_all["ci95_primary_high_pct"]), 0.0)
        self.assertNotIn("paired_t_p_effect_pct_full723", conus_all)

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
        selected = {
            row["family"]: row
            for row in rows
            if row["method"] == "heldout80"
        }
        self.assertAlmostEqual(float(selected["surface"]["mean_effect_pct"]), -13.4992, places=3)
        self.assertAlmostEqual(float(selected["aircraft"]["mean_effect_pct"]), -8.0574, places=3)
        self.assertTrue(
            all(float(row["ci95_high_pct"]) < 0.0 for row in selected.values())
        )

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
            (
                REPO_ROOT
                / "analysis/manuscript_tables/frozen2019_full723_paired_ttests.tex"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
