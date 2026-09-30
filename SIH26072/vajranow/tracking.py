from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackedCell:
    cell_id: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    dbz: float
    mean_dbz: float
    area_pixels: int
    area_km2: float
    age_minutes: int
    strikes: int
    strike_rate: float
    ir_cooling_rate: float
    dbz_trend: float
    history: list[tuple[float, float]] = field(default_factory=list)


def detect_cells(
    reflectivity: np.ndarray,
    threshold_dbz: float = 32.0,
    min_pixels: int = 4,
    grid_km: float = 2.0,
) -> list[dict]:
    """Multi-threshold connected-component convective storm cell detection.

    Identifies localized convective cores exceeding threshold dBZ.
    Computes centroid, area, peak dBZ, and mean reflectivity.
    """
    if reflectivity.ndim != 2:
        raise ValueError("reflectivity must be a 2D array")

    labels, count = label(reflectivity >= threshold_dbz)
    cells = []
    for component_id in range(1, count + 1):
        ys, xs = np.where(labels == component_id)
        if len(xs) < min_pixels:
            continue
        vals = reflectivity[ys, xs]
        max_val = float(vals.max())
        mean_val = float(vals.mean())
        area_px = int(len(xs))
        radius_px = float(np.sqrt(area_px / np.pi))

        cells.append(
            {
                "x": float(xs.mean()),
                "y": float(ys.mean()),
                "area_pixels": area_px,
                "area_km2": float(area_px * (grid_km**2)),
                "max_dbz": max_val,
                "mean_dbz": mean_val,
                "radius": radius_px,
            }
        )
    return cells


def link_cells(
    previous: list[dict],
    current: list[dict],
    max_distance: float = 12.0,
    dt_minutes: float = 10.0,
) -> list[dict]:
    """Link cell detections across consecutive time steps using Hungarian matching.

    Uses advection-predicted position (x + vx*dt, y + vy*dt) when velocity is available.
    """
    if not previous or not current:
        return []

    distances = np.zeros((len(previous), len(current)), dtype=np.float32)
    for i, old in enumerate(previous):
        # Predicted centroid position
        pred_x = old["x"] + old.get("vx", 0.0) * (dt_minutes / 10.0)
        pred_y = old["y"] + old.get("vy", 0.0) * (dt_minutes / 10.0)
        for j, new in enumerate(current):
            dist = np.hypot(pred_x - new["x"], pred_y - new["y"])
            # Penalty for large difference in peak dBZ
            dbz_diff = abs(old.get("max_dbz", 35.0) - new.get("max_dbz", 35.0))
            distances[i, j] = dist + 0.1 * dbz_diff

    old_indices, new_indices = linear_sum_assignment(distances)
    matches = []
    for old, new in zip(old_indices, new_indices, strict=True):
        if distances[old, new] <= max_distance:
            # Estimate velocity
            vx = (current[new]["x"] - previous[old]["x"]) / (dt_minutes / 10.0)
            vy = (current[new]["y"] - previous[old]["y"]) / (dt_minutes / 10.0)
            matches.append(
                {
                    "previous_index": int(old),
                    "current_index": int(new),
                    "distance": float(distances[old, new]),
                    "vx": float(vx),
                    "vy": float(vy),
                }
            )
    return matches
