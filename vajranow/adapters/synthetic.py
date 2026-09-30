from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from vajranow.config import load_config
from vajranow.synthetic import generate_cells, reflectivity_field


class SyntheticBackend:
    """Deterministic offline source emitting aligned 10-minute fields on the demo grid."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

    def get(self, t0: datetime, t1: datetime) -> xr.Dataset:
        if t1 < t0:
            raise ValueError("t1 must be greater than or equal to t0")
        region = self.config["region"]
        step = self.config["region"]["step_minutes"]
        width, height = region["width"], region["height"]
        timestamps = []
        current = t0
        while current <= t1:
            timestamps.append(current)
            current += timedelta(minutes=step)
        if not timestamps:
            timestamps.append(t0)
        satellite_arrays = []
        lightning_arrays = []
        yy, xx = np.mgrid[:height, :width]
        radar_fields = [[], []]
        for index, _timestamp in enumerate(timestamps):
            minute = index * step
            field = reflectivity_field(width, height, minute)
            for radar_index, (center_x, center_y) in enumerate(((19, 30), (45, 32))):
                in_range = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (max(width, height) * 0.67) ** 2
                radar_fields[radar_index].append(np.where(in_range, field, np.nan))
            satellite_arrays.append(np.stack([290.0 - field * 0.30, 282.0 - field * 0.24]))
            lightning = np.zeros((height, width), dtype=np.float32)
            for cell in generate_cells(width, height, minute):
                if cell.strikes:
                    lightning += cell.strikes * np.exp(-((xx - cell.x) ** 2 + (yy - cell.y) ** 2) / 8.0)
            lightning_arrays.append(lightning)
        radar_arrays = [np.stack(frames).astype(np.float32) for frames in radar_fields]
        availability = np.stack([np.isfinite(radar) for radar in radar_arrays]).astype(np.uint8)
        stacked = np.stack(radar_arrays)
        composite = np.max(np.where(np.isfinite(stacked), stacked, -np.inf), axis=0)
        composite[~np.isfinite(composite)] = 0.0
        time_coords = np.array([np.datetime64(value.replace(tzinfo=None)) for value in timestamps])
        return xr.Dataset(
            data_vars={
                "composite_dbz": (("time", "y", "x"), composite),
                "radar_availability": (("radar", "time", "y", "x"), availability),
                "satellite_ir": (("time", "band", "y", "x"), np.stack(satellite_arrays)),
                "lightning_density": (("time", "y", "x"), np.stack(lightning_arrays)),
                "satellite_available": (("time",), np.ones(len(timestamps), dtype=np.uint8)),
                "lightning_available": (("time",), np.ones(len(timestamps), dtype=np.uint8)),
                "model_available": (("time",), np.zeros(len(timestamps), dtype=np.uint8)),
            },
            coords={
                "time": time_coords,
                "radar": ["synthetic-west", "synthetic-east"],
                "band": ["ir069-proxy", "ir107-proxy"],
                "y": np.arange(height),
                "x": np.arange(width),
            },
            attrs={"data_mode": "synthetic", "label": "prototype, not operational"},
        )
