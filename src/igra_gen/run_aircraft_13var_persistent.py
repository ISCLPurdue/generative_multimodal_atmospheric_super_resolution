"""Inference engine for the paper's radiosonde, aircraft, and surface interfaces.

Use the entry points in ``reproduction/`` to register and run the published
R, R+A, R+S, and R+A+S configurations.
"""

import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset

WORKDIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(WORKDIR)
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, WORKDIR)

from igra_gen.generating.factory import sampler_factory
from igra_gen.models.precond import EDMPrecond
from igra_gen.utils import io


ERA5_ROOT = os.environ.get("ERA5_ROOT", "")
HYDRA_CFG = os.environ.get("ATMOSPHERIC_PRIOR_HYDRA_CONFIG", "")
DEFAULT_CHECKPOINT = os.environ.get("ATMOSPHERIC_PRIOR_CHECKPOINT", "")
IGRA_PKL = os.environ.get("IGRA_PKL", "")
NUM_CHANNELS = 13
AIRCRAFT_ROOT = os.environ.get("AIRCRAFT_ROOT", "")
SURFACE_METAR_ROOT = os.environ.get("SURFACE_METAR_ROOT", "")

IGRA_VARIABLES = [
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

AIRCRAFT_VARIABLES = [
    "temperature_500",
    "temperature_850",
    "u_component_of_wind_500",
    "v_component_of_wind_500",
    "u_component_of_wind_850",
    "v_component_of_wind_850",
]

SURFACE_METAR_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

EXPERIMENTS: Dict[str, dict] = {}

SPATIAL_SUPPORTS = {
    "strict_conus": {
        "lat": (24.0, 50.0),
        "lon": (-125.0, -66.0),
        "description": "Strict CONUS box used as the primary regional metric target.",
    },
    "conus_buffer": {
        "lat": (20.0, 55.0),
        "lon": (-135.0, -55.0),
        "description": "CONUS plus surrounding buffer for boundary-adjacent aircraft observations.",
    },
    "north_america": {
        "lat": (10.0, 70.0),
        "lon": (-170.0, -50.0),
        "description": "Broad North America diagnostic box.",
    },
    "global": {
        "lat": (-90.0, 90.0),
        "lon": (-180.0, 180.0),
        "description": "All available aircraft observations.",
    },
}


def empty_channels(n_channels: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    ch_locs = [np.empty((0, 2), dtype=np.float32) for _ in range(n_channels)]
    ch_vals = [np.empty((0,), dtype=np.float32) for _ in range(n_channels)]
    return ch_locs, ch_vals


class PersistentFullPoolRunner:
    def __init__(
        self,
        output_root: str,
        ens: int = 16,
        seed: int = 17,
        num_steps: int = 50,
        sigma_min: float = 0.005,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_min: float = 0.01,
        S_max: float = 50.0,
        S_noise: float = 1.003,
        igra_pkl: str = IGRA_PKL,
        aircraft_root: str = AIRCRAFT_ROOT,
        surface_metar_root: str = SURFACE_METAR_ROOT,
        checkpoint: str = DEFAULT_CHECKPOINT,
        era5_root: str = ERA5_ROOT,
        hydra_cfg: str = HYDRA_CFG,
        num_channels: int = NUM_CHANNELS,
        likelihood_structure: str = "source_specific",
        std_igra: float = 5e-4,
        gamma_igra: float = 2e-6,
        lambda_igra: float = 1.0,
        std_aircraft: float = 5e-4,
        gamma_aircraft: float = 2e-5,
        lambda_aircraft: float = 0.4,
        std_surface: float = 1.25e-4,
        gamma_surface: float = 4e-5,
        lambda_surface: float = 0.4,
        era5_split: str = "test",
        calendar_year: int = 2020,
    ):
        self.output_root = output_root
        self.samples_root = os.path.join(output_root, "samples")
        self.ens = ens
        self.seed = seed
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.igra_pkl = igra_pkl
        self.aircraft_roots = {
            "aircraft_pressure_window_25hpa": aircraft_root,
        }
        self.surface_metar_root = surface_metar_root
        self.checkpoint = checkpoint
        self.era5_root = era5_root
        self.hydra_cfg = hydra_cfg
        self.num_channels = num_channels
        self.era5_split = era5_split
        self.calendar_year = int(calendar_year)
        if likelihood_structure not in {"single", "source_specific"}:
            raise ValueError(f"Unknown likelihood_structure={likelihood_structure}")
        self.likelihood_structure = likelihood_structure
        self.likelihood_kwargs = {
            "std_igra": std_igra,
            "gamma_igra": gamma_igra,
            "lambda_igra": lambda_igra,
            "std_aircraft": std_aircraft,
            "gamma_aircraft": gamma_aircraft,
            "lambda_aircraft": lambda_aircraft,
            "std_surface": std_surface,
            "gamma_surface": gamma_surface,
            "lambda_surface": lambda_surface,
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        io.log0(f"Persistent runner using device={self.device}")

        means = np.load(os.path.join(self.era5_root, "normalize_mean.npz"))
        stds = np.load(os.path.join(self.era5_root, "normalize_std.npz"))
        self.mean_2m = float(np.asarray(means["2m_temperature"]).reshape(-1)[0])
        self.std_2m = float(np.asarray(stds["2m_temperature"]).reshape(-1)[0])
        self.era5_lat = np.load(os.path.join(self.era5_root, "lat.npy")).astype(np.float32)
        self.era5_lon = np.load(os.path.join(self.era5_root, "lon.npy")).astype(np.float32)

        conf = OmegaConf.load(self.hydra_cfg)
        conf.data.dataset.root = self.era5_root
        conf.data.dataset.split = self.era5_split
        self.dataset: Dataset = instantiate(conf.data.dataset, _convert_="object")
        self.model_vars = list(self.dataset.variables[:-1])
        if len(self.model_vars) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} variables, got {len(self.model_vars)}: {self.model_vars}")
        self.temp_idx = self.model_vars.index("2m_temperature")
        self.channel_mean = {
            var: float(np.asarray(means[var]).reshape(-1)[0])
            for var in self.model_vars
        }
        self.channel_std = {
            var: float(np.asarray(stds[var]).reshape(-1)[0])
            for var in self.model_vars
        }

        self.net = EDMPrecond(
            model=conf.model,
            img_resolution=(128, 256),
            img_channels=self.num_channels,
            sigma_data=1,
            sigma_max=80,
            sigma_min=0.005,
            condition_channels=1,
        ).to(self.device).eval()

        io.log0(f"Loading checkpoint once: {checkpoint}")
        chkpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in chkpt["ema"].items()}
        if any(k.startswith("model.") for k in state_dict):
            self.net.load_state_dict(state_dict)
        else:
            self.net.model.load_state_dict(state_dict)

        self.sample_fn = sampler_factory(
            mode="edm_pos_sample",
            net=self.net,
            conditioning_type="multimodal" if self.likelihood_structure == "source_specific" else "igra",
            in_shape=(16, 32),
            target_shape=(128, 256),
            lat_path=os.path.join(self.era5_root, "lat.npy"),
            lon_path=os.path.join(self.era5_root, "lon.npy"),
        )
        self.in_shape = (1, self.num_channels, 128, 256)

        io.log0(f"Loading IGRA pkl once: {igra_pkl}")
        with open(igra_pkl, "rb") as f:
            self.igra_data = pickle.load(f)

    def timestep_to_datetime(self, timestep: int) -> datetime:
        return datetime(self.calendar_year, 1, 1) + timedelta(hours=6 * int(timestep))

    def load_igra_channels(
        self,
        timestep: int,
        variables: Optional[Iterable[str]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if timestep >= len(self.igra_data):
            raise IndexError(f"{self.igra_pkl} has only {len(self.igra_data)} timesteps; got t{timestep:04d}")
        base_query_locations, base_true_values = self.igra_data[timestep]
        base_query_locations = base_query_locations[0]
        base_true_values = base_true_values[0]
        ch_locs, ch_vals = empty_channels(len(self.model_vars))
        allowed = set(variables) if variables is not None else None
        for src_idx, var_name in enumerate(IGRA_VARIABLES):
            if allowed is not None and var_name not in allowed:
                continue
            dst_idx = self.model_vars.index(var_name)
            ch_locs[dst_idx] = np.asarray(base_query_locations[src_idx], dtype=np.float32)
            ch_vals[dst_idx] = np.asarray(base_true_values[src_idx], dtype=np.float32)
        return ch_locs, ch_vals

    def _nearest_era5_flat_cells(self, locs: np.ndarray) -> np.ndarray:
        lat_axis = self.era5_lat
        lon_axis = self.era5_lon
        dlat = float(np.median(np.diff(lat_axis)))
        dlon = float(np.median(np.diff(lon_axis)))
        locs = np.asarray(locs, dtype=np.float64)
        lat_idx = np.rint((locs[:, 0] - float(lat_axis[0])) / dlat).astype(np.int64)
        lon_idx = np.rint((np.mod(locs[:, 1], 360.0) - float(lon_axis[0])) / dlon).astype(np.int64)
        lat_idx = np.clip(lat_idx, 0, lat_axis.size - 1)
        lon_idx = np.mod(lon_idx, lon_axis.size)
        return lat_idx * lon_axis.size + lon_idx

    def _nearest_era5_indices(self, locs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lat_axis = self.era5_lat
        lon_axis = self.era5_lon
        dlat = float(np.median(np.diff(lat_axis)))
        dlon = float(np.median(np.diff(lon_axis)))
        locs = np.asarray(locs, dtype=np.float64)
        lat_idx = np.rint((locs[:, 0] - float(lat_axis[0])) / dlat).astype(np.int64)
        lon_idx = np.rint((np.mod(locs[:, 1], 360.0) - float(lon_axis[0])) / dlon).astype(np.int64)
        lat_idx = np.clip(lat_idx, 0, lat_axis.size - 1)
        lon_idx = np.mod(lon_idx, lon_axis.size)
        return lat_idx, lon_idx

    def _distance_to_nearest_cell_center_in_cell_units(self, locs: np.ndarray) -> np.ndarray:
        if len(locs) == 0:
            return np.empty((0,), dtype=np.float32)
        lat_idx, lon_idx = self._nearest_era5_indices(locs)
        dlat = float(np.median(np.diff(self.era5_lat)))
        dlon = float(np.median(np.diff(self.era5_lon)))
        locs = np.asarray(locs, dtype=np.float64)
        lat_diff = (locs[:, 0] - self.era5_lat[lat_idx]) / dlat
        lon360 = np.mod(locs[:, 1], 360.0)
        lon_diff_deg = ((lon360 - self.era5_lon[lon_idx] + 180.0) % 360.0) - 180.0
        lon_diff = lon_diff_deg / dlon
        return np.sqrt(lat_diff**2 + lon_diff**2).astype(np.float32)

    def _cell_balanced_weights(self, locs: np.ndarray) -> np.ndarray:
        if len(locs) == 0:
            return np.empty((0,), dtype=np.float32)
        flat_cell = self._nearest_era5_flat_cells(locs)
        unique, inverse, counts = np.unique(flat_cell, return_inverse=True, return_counts=True)
        del unique
        weights = 1.0 / counts[inverse].astype(np.float32)
        return weights.astype(np.float32)

    @staticmethod
    def _simple_weights(locs: np.ndarray) -> np.ndarray:
        return np.ones((len(locs),), dtype=np.float32)

    def _spatial_support_mask(self, locs: np.ndarray, support: str) -> np.ndarray:
        if support not in SPATIAL_SUPPORTS:
            raise ValueError(f"Unknown aircraft spatial support={support}; choices={sorted(SPATIAL_SUPPORTS)}")
        locs = np.asarray(locs)
        lat_lo, lat_hi = SPATIAL_SUPPORTS[support]["lat"]
        lon_lo, lon_hi = SPATIAL_SUPPORTS[support]["lon"]
        lon = ((locs[:, 1] + 180.0) % 360.0) - 180.0
        return (
            np.isfinite(locs[:, 0])
            & np.isfinite(lon)
            & (locs[:, 0] >= lat_lo)
            & (locs[:, 0] <= lat_hi)
            & (lon >= lon_lo)
            & (lon <= lon_hi)
        )

    @staticmethod
    def _source_filter_mask(products: np.ndarray, source_filter: Optional[str]) -> np.ndarray:
        if source_filter is None or source_filter == "combined":
            return np.ones(products.shape, dtype=bool)
        if source_filter == "acars":
            return products == "acars"
        raise ValueError("source_filter must be one of None/combined/acars")

    def aggregate_points_to_era5_grid(
        self,
        locs: np.ndarray,
        vals: np.ndarray,
        pressures_hpa: Optional[np.ndarray] = None,
        target_pressure_hpa: Optional[float] = None,
        aggregation: str = "equal",
        sigma_pressure_hpa: float = 15.0,
        sigma_distance_cell: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate point observations to one normalized value per ERA5 cell.

        aggregation:
            equal: simple mean within each ERA5 cell.
            pressure: Gaussian pressure-offset weights.
            distance: Gaussian distance-to-nearest-cell-center weights.
            pressure_distance: product of pressure and distance weights.
        """
        locs = np.asarray(locs, dtype=np.float64)
        vals = np.asarray(vals, dtype=np.float64)
        if pressures_hpa is not None:
            pressures_hpa = np.asarray(pressures_hpa, dtype=np.float64)
        shape = (self.era5_lat.size, self.era5_lon.size)
        if len(vals) == 0:
            return (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=bool),
                np.zeros(shape, dtype=np.int64),
            )
        good = np.isfinite(locs[:, 0]) & np.isfinite(locs[:, 1]) & np.isfinite(vals)
        if aggregation in {"pressure", "pressure_distance"}:
            if pressures_hpa is None or target_pressure_hpa is None:
                raise ValueError("Pressure-weighted aircraft aggregation requires pressures_hpa and target_pressure_hpa")
            good = good & np.isfinite(pressures_hpa)
        locs = locs[good]
        vals = vals[good]
        if pressures_hpa is not None:
            pressures_hpa = pressures_hpa[good]
        if len(vals) == 0:
            return (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=bool),
                np.zeros(shape, dtype=np.int64),
            )
        flat_cell = self._nearest_era5_flat_cells(locs)
        n_cells = self.era5_lat.size * self.era5_lon.size
        if aggregation == "equal":
            weights = np.ones_like(vals, dtype=np.float64)
        elif aggregation == "pressure":
            weights = np.exp(-0.5 * ((pressures_hpa - float(target_pressure_hpa)) / float(sigma_pressure_hpa)) ** 2)
        elif aggregation == "distance":
            dist = self._distance_to_nearest_cell_center_in_cell_units(locs)
            weights = np.exp(-0.5 * (dist / float(sigma_distance_cell)) ** 2)
        elif aggregation == "pressure_distance":
            dist = self._distance_to_nearest_cell_center_in_cell_units(locs)
            w_pressure = np.exp(-0.5 * ((pressures_hpa - float(target_pressure_hpa)) / float(sigma_pressure_hpa)) ** 2)
            w_distance = np.exp(-0.5 * (dist / float(sigma_distance_cell)) ** 2)
            weights = w_pressure * w_distance
        else:
            raise ValueError(f"Unknown aircraft grid aggregation={aggregation}")
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        sum_vals = np.bincount(flat_cell, weights=vals * weights, minlength=n_cells)
        sum_weights = np.bincount(flat_cell, weights=weights, minlength=n_cells)
        count = np.bincount(flat_cell, minlength=n_cells).astype(np.int64)
        mask = sum_weights > 0
        grid = np.zeros(n_cells, dtype=np.float32)
        grid[mask] = (sum_vals[mask] / sum_weights[mask]).astype(np.float32)
        return grid.reshape(shape), mask.reshape(shape), count.reshape(shape)

    def load_aircraft_measurements(
        self,
        obs_mode: str,
        timestep: int,
        source_filter: Optional[str] = None,
        spatial_support: str = "global",
        weighting: str = "cell_balanced",
        return_pressure: bool = False,
        data_sources: Optional[Iterable[int]] = None,
    ):
        root = self.aircraft_roots[obs_mode]
        path = os.path.join(root, "obs", f"madis_aircraft_13var_t{timestep:04d}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing aircraft obs file for {obs_mode} t{timestep:04d}: {path}")
        data = np.load(path, allow_pickle=False)
        ch_locs, ch_vals = empty_channels(len(self.model_vars))
        ch_weights = [np.empty((0,), dtype=np.float32) for _ in range(len(self.model_vars))]
        ch_pressures = [np.empty((0,), dtype=np.float32) for _ in range(len(self.model_vars))]
        counts: Dict[str, int] = {}
        cells: Dict[str, int] = {}
        source_counts: Dict[str, Dict[str, int]] = {}
        for var in AIRCRAFT_VARIABLES:
            dst_idx = self.model_vars.index(var)
            locs = np.asarray(data[f"{var}_locs"], dtype=np.float32)
            vals_physical = np.asarray(data[f"{var}_vals"], dtype=np.float32)
            pressure = np.asarray(data[f"{var}_pressure_hpa_alt_derived"], dtype=np.float32)
            products = np.asarray(data[f"{var}_source_product"])
            source_codes = np.asarray(data[f"{var}_data_source"], dtype=np.int16)
            keep = self._spatial_support_mask(locs, spatial_support) & self._source_filter_mask(products, source_filter)
            if data_sources is not None:
                allowed_sources = np.asarray(sorted(set(int(value) for value in data_sources)), dtype=np.int16)
                keep = keep & np.isin(source_codes, allowed_sources)
            locs = locs[keep]
            vals_physical = vals_physical[keep]
            pressure = pressure[keep]
            products = products[keep]
            source_codes = source_codes[keep]
            vals = ((vals_physical - self.channel_mean[var]) / self.channel_std[var]).astype(np.float32)
            ch_locs[dst_idx] = locs
            ch_vals[dst_idx] = vals
            ch_pressures[dst_idx] = pressure
            if weighting == "simple":
                ch_weights[dst_idx] = self._simple_weights(locs)
            elif weighting == "cell_balanced":
                ch_weights[dst_idx] = self._cell_balanced_weights(locs)
            else:
                raise ValueError(f"Unknown aircraft sparse weighting={weighting}")
            counts[var] = int(vals.size)
            cells[var] = int(np.unique(self._nearest_era5_flat_cells(locs)).size) if vals.size else 0
            source_counts[var] = {
                "acars": int(np.sum(products == "acars")) if vals.size else 0,
                **{f"dataSource_{int(code)}": int(np.sum(source_codes == code)) for code in np.unique(source_codes)},
            }
        metadata_json = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
        meta = {
            "obs_mode": obs_mode,
            "aircraft_root": root,
            "aircraft_file": path,
            "aircraft_source_filter": source_filter or "combined",
            "aircraft_data_sources": "all" if data_sources is None else sorted(set(int(value) for value in data_sources)),
            "aircraft_spatial_support": spatial_support,
            "aircraft_spatial_support_json": json.dumps(SPATIAL_SUPPORTS[spatial_support], sort_keys=True),
            "aircraft_counts_json": json.dumps(counts, sort_keys=True),
            "aircraft_source_counts_json": json.dumps(source_counts, sort_keys=True),
            "aircraft_covered_era5_cells_json": json.dumps(cells, sort_keys=True),
            "source_metadata_json": metadata_json,
        }
        if return_pressure:
            return meta, ch_locs, ch_vals, ch_weights, ch_pressures
        return meta, ch_locs, ch_vals, ch_weights

    def load_aircraft_cell_mean_grid(
        self,
        obs_mode: str,
        timestep: int,
        source_filter: Optional[str] = None,
        spatial_support: str = "global",
        aggregation: str = "equal",
        sigma_pressure_hpa: float = 15.0,
        sigma_distance_cell: float = 1.0,
        data_sources: Optional[Iterable[int]] = None,
    ) -> Tuple[Dict[str, str], List[np.ndarray], List[np.ndarray], List[int], np.ndarray]:
        meta, ch_locs, ch_vals, _, ch_pressures = self.load_aircraft_measurements(
            obs_mode,
            timestep,
            source_filter=source_filter,
            spatial_support=spatial_support,
            return_pressure=True,
            data_sources=data_sources,
        )
        grids: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        counts: List[np.ndarray] = []
        channel_indices: List[int] = []
        valid_cells: Dict[str, int] = {}
        max_points: Dict[str, int] = {}
        for var in AIRCRAFT_VARIABLES:
            idx = self.model_vars.index(var)
            target_pressure_hpa = 500.0 if var.endswith("_500") else 850.0
            grid, mask, count = self.aggregate_points_to_era5_grid(
                ch_locs[idx],
                ch_vals[idx],
                pressures_hpa=ch_pressures[idx],
                target_pressure_hpa=target_pressure_hpa,
                aggregation=aggregation,
                sigma_pressure_hpa=sigma_pressure_hpa,
                sigma_distance_cell=sigma_distance_cell,
            )
            grids.append(grid)
            masks.append(mask)
            counts.append(count)
            channel_indices.append(idx)
            valid_cells[var] = int(mask.sum())
            max_points[var] = int(count.max()) if count.size else 0
        count_stack = np.stack(counts, axis=0)
        meta.update({
            "obs_space": "aircraft_cell_mean_grid",
            "aircraft_grid_aggregation": aggregation,
            "aircraft_pressure_weight_sigma_hpa": str(sigma_pressure_hpa),
            "aircraft_distance_weight_sigma_cell": str(sigma_distance_cell),
            "aircraft_valid_era5_cells_json": json.dumps(valid_cells, sort_keys=True),
            "aircraft_max_points_per_cell_json": json.dumps(max_points, sort_keys=True),
        })
        return meta, grids, masks, channel_indices, count_stack

    def load_surface_metar_measurements(
        self,
        timestep: int,
        variables: Optional[Iterable[str]] = None,
        weighting: str = "cell_balanced",
        spatial_support: str = "strict_conus",
    ) -> Tuple[Dict[str, str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        path = os.path.join(self.surface_metar_root, "obs", f"madis_metar_surface_13var_t{timestep:04d}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing METAR surface obs file for t{timestep:04d}: {path}")
        data = np.load(path, allow_pickle=False)
        ch_locs, ch_vals = empty_channels(len(self.model_vars))
        ch_weights = [np.empty((0,), dtype=np.float32) for _ in range(len(self.model_vars))]
        counts: Dict[str, int] = {}
        cells: Dict[str, int] = {}
        allowed = set(variables) if variables is not None else set(SURFACE_METAR_VARIABLES)
        for var in SURFACE_METAR_VARIABLES:
            if var not in allowed:
                continue
            dst_idx = self.model_vars.index(var)
            locs = np.asarray(data[f"{var}_locs"], dtype=np.float32)
            vals_physical = np.asarray(data[f"{var}_vals"], dtype=np.float32)
            keep = self._spatial_support_mask(locs, spatial_support)
            locs = locs[keep]
            vals_physical = vals_physical[keep]
            vals = ((vals_physical - self.channel_mean[var]) / self.channel_std[var]).astype(np.float32)
            ch_locs[dst_idx] = locs
            ch_vals[dst_idx] = vals
            if weighting == "simple":
                ch_weights[dst_idx] = self._simple_weights(locs)
            elif weighting == "cell_balanced":
                ch_weights[dst_idx] = self._cell_balanced_weights(locs)
            else:
                raise ValueError(f"Unknown surface sparse weighting={weighting}")
            counts[var] = int(vals.size)
            cells[var] = int(np.unique(self._nearest_era5_flat_cells(locs)).size) if vals.size else 0
        metadata_json = str(data["metadata_json"]) if "metadata_json" in data.files else "{}"
        meta = {
            "obs_mode": "surface_metar",
            "surface_metar_root": self.surface_metar_root,
            "surface_metar_file": path,
            "surface_spatial_support": spatial_support,
            "surface_spatial_support_json": json.dumps(SPATIAL_SUPPORTS[spatial_support], sort_keys=True),
            "surface_variables_json": json.dumps(sorted(allowed)),
            "surface_counts_json": json.dumps(counts, sort_keys=True),
            "surface_covered_era5_cells_json": json.dumps(cells, sort_keys=True),
            "source_metadata_json": metadata_json,
        }
        return meta, ch_locs, ch_vals, ch_weights

    def load_surface_metar_cell_mean_grid(
        self,
        timestep: int,
        variables: Optional[Iterable[str]] = None,
        aggregation: str = "equal",
        spatial_support: str = "strict_conus",
    ) -> Tuple[Dict[str, str], List[np.ndarray], List[np.ndarray], List[int], np.ndarray]:
        meta, ch_locs, ch_vals, _ = self.load_surface_metar_measurements(
            timestep,
            variables=variables,
            weighting="cell_balanced",
            spatial_support=spatial_support,
        )
        grids: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        counts: List[np.ndarray] = []
        channel_indices: List[int] = []
        valid_cells: Dict[str, int] = {}
        max_points: Dict[str, int] = {}
        allowed = set(variables) if variables is not None else set(SURFACE_METAR_VARIABLES)
        for var in SURFACE_METAR_VARIABLES:
            if var not in allowed:
                continue
            idx = self.model_vars.index(var)
            grid, mask, count = self.aggregate_points_to_era5_grid(
                ch_locs[idx],
                ch_vals[idx],
                aggregation=aggregation,
            )
            grids.append(grid)
            masks.append(mask)
            counts.append(count)
            channel_indices.append(idx)
            valid_cells[var] = int(mask.sum())
            max_points[var] = int(count.max()) if count.size else 0
        count_stack = np.stack(counts, axis=0) if counts else np.zeros((0,), dtype=np.int64)
        meta.update({
            "obs_space": "surface_cell_mean_grid",
            "surface_grid_aggregation": aggregation,
            "surface_valid_era5_cells_json": json.dumps(valid_cells, sort_keys=True),
            "surface_max_points_per_cell_json": json.dumps(max_points, sort_keys=True),
        })
        return meta, grids, masks, channel_indices, count_stack

    def load_observation_measurements(self, obs_mode: str, timestep: int) -> Tuple[Dict[str, str], np.ndarray, np.ndarray]:
        if obs_mode == "none":
            locs = np.empty((0, 2), dtype=np.float32)
            vals = np.empty((0,), dtype=np.float32)
            return {"obs_mode": "none", "kept_points": "0"}, locs, vals
        raise ValueError(f"Unknown obs_mode={obs_mode}")

    def output_path(self, experiment: str, timestep: int) -> str:
        return os.path.join(
            self.samples_root,
            experiment,
            f"{experiment}_t{timestep:04d}_e{self.ens}_s{self.num_steps}.npy",
        )

    def as_ensemble_measurement(self, ch_locs: List[np.ndarray], ch_vals: List[np.ndarray]):
        return [[ch_locs] * self.ens, [ch_vals] * self.ens]

    def as_weighted_ensemble_measurement(
        self,
        ch_locs: List[np.ndarray],
        ch_vals: List[np.ndarray],
        ch_weights: List[np.ndarray],
    ):
        return {
            "kind": "weighted_sparse",
            "query_locations": [ch_locs] * self.ens,
            "true_values": [ch_vals] * self.ens,
            "weights": [ch_weights] * self.ens,
        }

    def run_one(self, experiment: str, timestep: int, overwrite: bool = False) -> str:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment={experiment}. Choices: {sorted(EXPERIMENTS)}")
        spec = EXPERIMENTS[experiment]
        output = self.output_path(experiment, timestep)
        if os.path.exists(output) and not overwrite:
            io.log0(f"Skipping existing output: {output}")
            return output

        target_dt = self.timestep_to_datetime(timestep)
        io.log0(f"Running {experiment} t{timestep:04d} target_dt={target_dt.isoformat()}")
        obs_space = spec.get("obs_space", "points")
        obs_mask = obs_count = None
        aircraft_locs = aircraft_vals = aircraft_weights = None
        aircraft_grids = aircraft_masks = aircraft_channel_indices = None
        surface_locs = surface_vals = surface_weights = None
        surface_grids = surface_masks = surface_channel_indices = None
        if obs_space == "aircraft_surface_sparse":
            aircraft_weighting = spec.get("aircraft_weighting", "simple")
            surface_weighting = spec.get("surface_weighting", "simple")
            aircraft_meta, aircraft_locs, aircraft_vals, aircraft_weights = self.load_aircraft_measurements(
                spec["obs_mode"],
                timestep,
                source_filter=spec.get("aircraft_source_filter"),
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                weighting=aircraft_weighting,
                data_sources=spec.get("aircraft_data_sources"),
            )
            surface_meta, surface_locs, surface_vals, surface_weights = self.load_surface_metar_measurements(
                timestep,
                variables=spec.get("surface_variables"),
                weighting=surface_weighting,
                spatial_support=spec.get("surface_spatial_support", "strict_conus"),
            )
            obs_meta = {
                "obs_space": "aircraft_surface_sparse",
                "aircraft_sparse_weighting": aircraft_weighting,
                "surface_sparse_weighting": surface_weighting,
                "aircraft_meta": aircraft_meta,
                "surface_meta": surface_meta,
            }
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space in {"aircraft_weighted_sparse", "aircraft_simple_sparse"}:
            obs_meta, aircraft_locs, aircraft_vals, aircraft_weights = self.load_aircraft_measurements(
                spec["obs_mode"],
                timestep,
                source_filter=spec.get("aircraft_source_filter"),
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                weighting="simple" if obs_space == "aircraft_simple_sparse" else "cell_balanced",
                data_sources=spec.get("aircraft_data_sources"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space == "aircraft_cell_mean_grid":
            obs_meta, aircraft_grids, aircraft_masks, aircraft_channel_indices, obs_count = self.load_aircraft_cell_mean_grid(
                spec["obs_mode"],
                timestep,
                source_filter=spec.get("aircraft_source_filter"),
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                aggregation=spec.get("aircraft_grid_aggregation", "equal"),
                sigma_pressure_hpa=float(spec.get("aircraft_pressure_weight_sigma_hpa", 15.0)),
                sigma_distance_cell=float(spec.get("aircraft_distance_weight_sigma_cell", 1.0)),
                data_sources=spec.get("aircraft_data_sources"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space in {"surface_weighted_sparse", "surface_simple_sparse"}:
            obs_meta, surface_locs, surface_vals, surface_weights = self.load_surface_metar_measurements(
                timestep,
                variables=spec.get("surface_variables"),
                weighting="simple" if obs_space == "surface_simple_sparse" else "cell_balanced",
                spatial_support=spec.get("surface_spatial_support", "strict_conus"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space == "surface_cell_mean_grid":
            obs_meta, surface_grids, surface_masks, surface_channel_indices, obs_count = self.load_surface_metar_cell_mean_grid(
                timestep,
                variables=spec.get("surface_variables"),
                aggregation=spec.get("surface_grid_aggregation", "equal"),
                spatial_support=spec.get("surface_spatial_support", "strict_conus"),
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        elif obs_space == "aircraft_surface_cell_mean_grid":
            aircraft_meta, aircraft_grids, aircraft_masks, aircraft_channel_indices, aircraft_count = self.load_aircraft_cell_mean_grid(
                spec["obs_mode"],
                timestep,
                source_filter=spec.get("aircraft_source_filter"),
                spatial_support=spec.get("aircraft_spatial_support", "global"),
                aggregation=spec.get("aircraft_grid_aggregation", "equal"),
                sigma_pressure_hpa=float(spec.get("aircraft_pressure_weight_sigma_hpa", 15.0)),
                sigma_distance_cell=float(spec.get("aircraft_distance_weight_sigma_cell", 1.0)),
                data_sources=spec.get("aircraft_data_sources"),
            )
            surface_meta, surface_grids, surface_masks, surface_channel_indices, surface_count = self.load_surface_metar_cell_mean_grid(
                timestep,
                variables=spec.get("surface_variables"),
                aggregation=spec.get("surface_grid_aggregation", "equal"),
                spatial_support=spec.get("surface_spatial_support", "strict_conus"),
            )
            obs_meta = {
                "obs_space": "aircraft_surface_cell_mean_grid",
                "aircraft_meta": aircraft_meta,
                "surface_meta": surface_meta,
            }
            obs_count = np.asarray(
                list(np.ravel(aircraft_count)) + list(np.ravel(surface_count)),
                dtype=np.int64,
            )
            obs_locs = np.empty((0, 2), dtype=np.float32)
            obs_vals = np.empty((0,), dtype=np.float32)
        else:
            obs_meta, obs_locs, obs_vals = self.load_observation_measurements(spec["obs_mode"], timestep)

        if spec["use_igra"]:
            igra_locs, igra_vals = self.load_igra_channels(timestep, variables=spec.get("igra_variables"))
        else:
            igra_locs, igra_vals = empty_channels(len(self.model_vars))

        if self.likelihood_structure == "single":
            ch_locs = [np.asarray(x, dtype=np.float32) for x in igra_locs]
            ch_vals = [np.asarray(x, dtype=np.float32) for x in igra_vals]
            ch_locs[self.temp_idx] = np.concatenate([ch_locs[self.temp_idx], obs_locs], axis=0)
            ch_vals[self.temp_idx] = np.concatenate([ch_vals[self.temp_idx], obs_vals], axis=0)
            measurement = self.as_ensemble_measurement(ch_locs, ch_vals)
        else:
            measurement = {}
            ch_locs, ch_vals = empty_channels(len(self.model_vars))
            if spec["use_igra"]:
                measurement["igra"] = self.as_ensemble_measurement(igra_locs, igra_vals)
                ch_locs = [np.asarray(x, dtype=np.float32) for x in igra_locs]
                ch_vals = [np.asarray(x, dtype=np.float32) for x in igra_vals]
            if obs_space in {"aircraft_weighted_sparse", "aircraft_simple_sparse"}:
                obs_modality = spec["obs_modality"]
                if obs_modality != "aircraft":
                    raise ValueError(f"Experiment {experiment} has aircraft obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = self.as_weighted_ensemble_measurement(
                    aircraft_locs,
                    aircraft_vals,
                    aircraft_weights,
                )
                for i in range(len(self.model_vars)):
                    if len(aircraft_vals[i]):
                        ch_locs[i] = np.concatenate([ch_locs[i], aircraft_locs[i]], axis=0)
                        ch_vals[i] = np.concatenate([ch_vals[i], aircraft_vals[i]], axis=0)
            if obs_space == "aircraft_surface_sparse":
                if spec["obs_modality"] != "aircraft_surface":
                    raise ValueError(
                        f"Experiment {experiment} has aircraft+surface sparse obs "
                        f"but invalid obs_modality={spec['obs_modality']}"
                    )
                measurement["aircraft"] = self.as_weighted_ensemble_measurement(
                    aircraft_locs,
                    aircraft_vals,
                    aircraft_weights,
                )
                measurement["surface"] = self.as_weighted_ensemble_measurement(
                    surface_locs,
                    surface_vals,
                    surface_weights,
                )
                for i in range(len(self.model_vars)):
                    if len(aircraft_vals[i]):
                        ch_locs[i] = np.concatenate([ch_locs[i], aircraft_locs[i]], axis=0)
                        ch_vals[i] = np.concatenate([ch_vals[i], aircraft_vals[i]], axis=0)
                    if len(surface_vals[i]):
                        ch_locs[i] = np.concatenate([ch_locs[i], surface_locs[i]], axis=0)
                        ch_vals[i] = np.concatenate([ch_vals[i], surface_vals[i]], axis=0)
            if obs_space == "aircraft_cell_mean_grid":
                obs_modality = spec["obs_modality"]
                if obs_modality != "aircraft":
                    raise ValueError(f"Experiment {experiment} has aircraft grid obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = {
                    "kind": "multi_grid",
                    "grids": aircraft_grids,
                    "masks": aircraft_masks,
                    "channel_indices": aircraft_channel_indices,
                }
            if obs_space == "aircraft_surface_cell_mean_grid":
                if spec["obs_modality"] != "aircraft_surface":
                    raise ValueError(
                        f"Experiment {experiment} has aircraft+surface grid obs "
                        f"but invalid obs_modality={spec['obs_modality']}"
                    )
                measurement["aircraft"] = {
                    "kind": "multi_grid",
                    "grids": aircraft_grids,
                    "masks": aircraft_masks,
                    "channel_indices": aircraft_channel_indices,
                }
                measurement["surface"] = {
                    "kind": "multi_grid",
                    "grids": surface_grids,
                    "masks": surface_masks,
                    "channel_indices": surface_channel_indices,
                }
            if obs_space in {"surface_weighted_sparse", "surface_simple_sparse"}:
                obs_modality = spec["obs_modality"]
                if obs_modality not in {"surface", "metar"}:
                    raise ValueError(f"Experiment {experiment} has surface obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = self.as_weighted_ensemble_measurement(
                    surface_locs,
                    surface_vals,
                    surface_weights,
                )
                for i in range(len(self.model_vars)):
                    if len(surface_vals[i]):
                        ch_locs[i] = np.concatenate([ch_locs[i], surface_locs[i]], axis=0)
                        ch_vals[i] = np.concatenate([ch_vals[i], surface_vals[i]], axis=0)
            if obs_space == "surface_cell_mean_grid":
                obs_modality = spec["obs_modality"]
                if obs_modality not in {"surface", "metar"}:
                    raise ValueError(f"Experiment {experiment} has surface grid obs but invalid obs_modality={obs_modality}")
                measurement[obs_modality] = {
                    "kind": "multi_grid",
                    "grids": surface_grids,
                    "masks": surface_masks,
                    "channel_indices": surface_channel_indices,
                }
            if not measurement:
                measurement = None

        nonempty = [(self.model_vars[i], len(v)) for i, v in enumerate(ch_vals) if len(v)]
        if obs_space == "aircraft_cell_mean_grid" and aircraft_masks is not None:
            grid_valid_cells = int(sum(mask.sum() for mask in aircraft_masks))
        elif obs_space == "surface_cell_mean_grid" and surface_masks is not None:
            grid_valid_cells = int(sum(mask.sum() for mask in surface_masks))
        elif obs_space == "aircraft_surface_sparse":
            aircraft_cells = json.loads(aircraft_meta.get("aircraft_covered_era5_cells_json", "{}"))
            surface_cells = json.loads(surface_meta.get("surface_covered_era5_cells_json", "{}"))
            grid_valid_cells = int(sum(int(v) for v in aircraft_cells.values()))
            grid_valid_cells += int(sum(int(v) for v in surface_cells.values()))
        elif obs_space == "aircraft_surface_cell_mean_grid":
            grid_valid_cells = int(sum(mask.sum() for mask in aircraft_masks))
            grid_valid_cells += int(sum(mask.sum() for mask in surface_masks))
        else:
            grid_valid_cells = int(obs_mask.sum()) if obs_mask is not None else 0
        grid_native_points = 0
        if obs_space in {"aircraft_weighted_sparse", "aircraft_simple_sparse", "aircraft_cell_mean_grid"}:
            aircraft_counts = json.loads(obs_meta.get("aircraft_counts_json", "{}"))
            grid_native_points = int(sum(int(v) for v in aircraft_counts.values()))
        elif obs_space in {"surface_weighted_sparse", "surface_simple_sparse", "surface_cell_mean_grid"}:
            surface_counts = json.loads(obs_meta.get("surface_counts_json", "{}"))
            grid_native_points = int(sum(int(v) for v in surface_counts.values()))
        elif obs_space == "aircraft_surface_sparse":
            aircraft_counts = json.loads(aircraft_meta.get("aircraft_counts_json", "{}"))
            surface_counts = json.loads(surface_meta.get("surface_counts_json", "{}"))
            grid_native_points = int(sum(int(v) for v in aircraft_counts.values()))
            grid_native_points += int(sum(int(v) for v in surface_counts.values()))
        elif obs_space == "aircraft_surface_cell_mean_grid":
            aircraft_counts = json.loads(obs_meta["aircraft_meta"].get("aircraft_counts_json", "{}"))
            surface_counts = json.loads(obs_meta["surface_meta"].get("surface_counts_json", "{}"))
            grid_native_points = int(sum(int(v) for v in aircraft_counts.values()))
            grid_native_points += int(sum(int(v) for v in surface_counts.values()))
        io.log0(
            f"Observation summary | experiment={experiment} obs_points={len(obs_vals)} "
            f"grid_valid_cells={grid_valid_cells} grid_native_points={grid_native_points} "
            f"likelihood_structure={self.likelihood_structure} nonempty_channels={len(nonempty)} "
            f"total_condition_points={sum(n for _, n in nonempty)}"
        )

        condition, _ = self.dataset.__getitem__(timestep)
        condition = condition.float()[None, :].to(self.device)

        outs = []
        for i in range(self.ens):
            generator = torch.Generator(device=self.device).manual_seed(self.seed + i)
            sample = self.sample_fn(
                measurement,
                generator,
                condition=condition,
                in_shape=self.in_shape,
                device=self.device,
                num_steps=self.num_steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                S_churn=self.S_churn,
                S_min=self.S_min,
                S_max=self.S_max,
                S_noise=self.S_noise,
                **self.likelihood_kwargs,
            ).detach().cpu().numpy()
            outs.append(sample.squeeze())
            io.log0(f"Completed {experiment} t{timestep:04d} sample {i + 1}/{self.ens}")

        out_array = np.asarray(outs, dtype=np.float32)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        np.save(output, out_array)

        meta_output = output[:-4] + "_measurement_summary.npz"
        np.savez_compressed(
            meta_output,
            timestep=np.asarray(timestep, dtype=np.int64),
            target_datetime=np.asarray(target_dt.isoformat()),
            experiment=np.asarray(experiment),
            description=np.asarray(spec["description"]),
            use_igra=np.asarray(spec["use_igra"]),
            obs_mode=np.asarray(spec["obs_mode"]),
            ens=np.asarray(self.ens, dtype=np.int64),
            num_steps=np.asarray(self.num_steps, dtype=np.int64),
            seed=np.asarray(self.seed, dtype=np.int64),
            checkpoint=np.asarray(self.checkpoint),
            igra_pkl=np.asarray(self.igra_pkl if spec["use_igra"] else ""),
            channel_names=np.asarray(self.model_vars),
            channel_counts=np.asarray([len(v) for v in ch_vals], dtype=np.int64),
            obs_points=np.asarray(len(obs_vals), dtype=np.int64),
            obs_space=np.asarray(obs_space),
            grid_valid_cells=np.asarray(grid_valid_cells, dtype=np.int64),
            grid_native_points=np.asarray(grid_native_points, dtype=np.int64),
            grid_count=np.asarray(obs_count if obs_count is not None else np.zeros((0,), dtype=np.int64), dtype=np.int64),
            grid_mask=np.asarray(obs_mask if obs_mask is not None else np.zeros((0,), dtype=bool), dtype=bool),
            wind_channel_indices=np.asarray(aircraft_channel_indices if aircraft_channel_indices is not None else np.zeros((0,), dtype=np.int64), dtype=np.int64),
            wind_grid_masks=np.asarray(aircraft_masks if aircraft_masks is not None else np.zeros((0,), dtype=bool), dtype=bool),
            surface_channel_indices=np.asarray(surface_channel_indices if surface_channel_indices is not None else np.zeros((0,), dtype=np.int64), dtype=np.int64),
            surface_grid_masks=np.asarray(surface_masks if surface_masks is not None else np.zeros((0,), dtype=bool), dtype=bool),
            obs_meta_json=np.asarray(json.dumps(obs_meta, sort_keys=True)),
            likelihood_structure=np.asarray(self.likelihood_structure),
            likelihood_kwargs_json=np.asarray(json.dumps(self.likelihood_kwargs, sort_keys=True)),
        )
        io.log0(f"Saved samples to {output} shape={out_array.shape}")
        io.log0(f"Saved measurement summary to {meta_output}")
        return output

    def run_many(self, experiments: Iterable[str], timesteps: Iterable[int], overwrite: bool = False) -> None:
        for timestep in timesteps:
            for experiment in experiments:
                self.run_one(experiment=experiment, timestep=int(timestep), overwrite=overwrite)
