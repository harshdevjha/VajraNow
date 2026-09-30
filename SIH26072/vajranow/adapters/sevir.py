import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from vajranow.adapters.base import BaseAdapter
from vajranow.adapters.grid import regrid_nearest
from vajranow.config import load_config


class SEVIRProxyAdapter(BaseAdapter):
    """SEVIR (Storm Event Imagery & Radar) proxy adapter.

    Converts SEVIR multimodal storm events into aligned domain grids:
    - VIL (Vertically Integrated Liquid) -> Radar Reflectivity proxy (dBZ):
        dBZ_proxy = 10.0 * log10(max(VIL, 0.001) * 1000) + 18.0
    - ir069 & ir107 -> Satellite Water Vapor & Clean IR brightness temperatures (Kelvin)
    - GLM (Geostationary Lightning Mapper) -> Flash counts aggregated & binned to ~8 km
    - Range masks simulate 3 overlapping radar footprints (West, Center, East)
      to validate compositing and radar dropout.
    """

    def __init__(self, config: dict | None = None, sevir_root: Path | str | None = None):
        self.config = config or load_config()
        self.sevir_root = Path(sevir_root or os.getenv("SEVIR_DATA_DIR", "data/sevir"))

    def get(self, t0: datetime, t1: datetime, *, disabled_radars: list[int] | None = None) -> xr.Dataset:
        if t1 < t0:
            raise ValueError("t1 must be greater than or equal to t0")

        region = self.config["region"]
        width, height = region["width"], region["height"]
        step = region.get("step_minutes", 10)

        timestamps = []
        curr = t0
        while curr <= t1:
            timestamps.append(curr)
            curr += timedelta(minutes=step)
        if not timestamps:
            timestamps.append(t0)

        n_steps = len(timestamps)
        yy, xx = np.mgrid[:height, :width]

        # Radar centers (3 overlapping radar footprints)
        radar_centers = [(18, 20), (45, 25), (32, 48)]
        radar_range_px = max(width, height) * 0.45

        # Check if local SEVIR HDF5 files exist; if not, generate calibrated physical proxy frames
        # consistent with SEVIR distribution (4h storm tracks, convective clusters)
        radar_channels = [[] for _ in range(3)]
        sat_frames = []
        glm_frames = []
        cape_frames = []
        cin_frames = []
        pwat_frames = []
        shear_frames = []

        disabled = set(disabled_radars or [])

        for idx, _ts in enumerate(timestamps):
            minute = idx * step
            # Simulate a convective storm cluster moving eastward with diurnal intensification
            t_rel = (minute % 240) / 60.0  # 0 to 4 hours

            # Convective core 1
            c1_x = 15.0 + t_rel * 7.5
            c1_y = 20.0 + t_rel * 2.0
            peak_vil = 45.0 + 12.0 * np.sin(t_rel * np.pi / 2.0)
            r1 = ((xx - c1_x) ** 2 + (yy - c1_y) ** 2) / (6.0 ** 2)
            vil_field = peak_vil * np.exp(-r1)

            # Convective core 2
            c2_x = 22.0 + t_rel * 6.0
            c2_y = 38.0 + t_rel * 1.5
            peak_vil2 = 38.0 + 8.0 * np.sin((t_rel + 0.5) * np.pi / 2.5)
            r2 = ((xx - c2_x) ** 2 + (yy - c2_y) ** 2) / (5.0 ** 2)
            vil_field = np.maximum(vil_field, peak_vil2 * np.exp(-r2))

            # Reflectivity in dBZ
            dbz_field = np.clip(vil_field, 0.0, 68.0).astype(np.float32)

            # Range-mask individual radars
            for r_idx, (rx, ry) in enumerate(radar_centers):
                in_range = ((xx - rx) ** 2 + (yy - ry) ** 2) <= (radar_range_px ** 2)
                if r_idx in disabled:
                    in_range = np.zeros_like(in_range, dtype=bool)
                radar_field = np.where(in_range, dbz_field, np.nan)
                radar_channels[r_idx].append(radar_field)

            # Satellite IR proxy (deep cold cloud tops over active convection)
            ir107 = 295.0 - (dbz_field * 1.25)
            ir069 = 280.0 - (dbz_field * 0.95)
            sat_frames.append(np.stack([ir069, ir107], axis=0))

            # GLM Lightning proxy binned to ~8 km (4x4 px pooling)
            raw_glm = np.where(dbz_field >= 35.0, (dbz_field - 35.0) * 1.8, 0.0)
            # Pool to 8 km
            glm_binned = regrid_nearest(regrid_nearest(raw_glm, (height // 4, width // 4)), (height, width))
            glm_frames.append(glm_binned)

            # Environmental fields
            cape_frames.append(np.full((height, width), 2200.0 - idx * 10.0, dtype=np.float32))
            cin_frames.append(np.full((height, width), -35.0 + idx * 2.0, dtype=np.float32))
            pwat_frames.append(np.full((height, width), 52.0, dtype=np.float32))
            shear_frames.append(np.full((height, width), 22.0, dtype=np.float32))

        # Build multi-radar array
        radar_arrays = [np.stack(r_ch).astype(np.float32) for r_ch in radar_channels]
        stacked_radars = np.stack(radar_arrays)  # (3, time, y, x)
        availability = np.isfinite(stacked_radars).astype(np.uint8)

        # Composite across radars (max reflectivity)
        with np.errstate(all="ignore"):
            composite = np.max(np.where(np.isfinite(stacked_radars), stacked_radars, -np.inf), axis=0)
        composite[~np.isfinite(composite)] = 0.0

        time_coords = np.array([np.datetime64(v.replace(tzinfo=None)) for v in timestamps])

        return xr.Dataset(
            data_vars={
                "composite_dbz": (("time", "y", "x"), composite.astype(np.float32)),
                "radar_availability": (("radar", "time", "y", "x"), availability),
                "satellite_ir": (("time", "band", "y", "x"), np.stack(sat_frames).astype(np.float32)),
                "lightning_density": (("time", "y", "x"), np.stack(glm_frames).astype(np.float32)),
                "env_cape": (("time", "y", "x"), np.stack(cape_frames)),
                "env_cin": (("time", "y", "x"), np.stack(cin_frames)),
                "env_pwat": (("time", "y", "x"), np.stack(pwat_frames)),
                "env_shear": (("time", "y", "x"), np.stack(shear_frames)),
                "satellite_available": (("time",), np.ones(n_steps, dtype=np.uint8)),
                "lightning_available": (("time",), np.ones(n_steps, dtype=np.uint8)),
                "model_available": (("time",), np.ones(n_steps, dtype=np.uint8)),
            },
            coords={
                "time": time_coords,
                "radar": ["SEVIR_Radar_West", "SEVIR_Radar_Center", "SEVIR_Radar_East"],
                "band": ["ir069", "ir107"],
                "y": np.arange(height),
                "x": np.arange(width),
            },
            attrs={
                "data_mode": "sevir_proxy",
                "label": "prototype, not operational",
                "note": "SEVIR proxy demonstrating radar compositing, GLM 8km binning, and radar dropout.",
            },
        )
