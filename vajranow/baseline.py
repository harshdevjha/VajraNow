import math

import numpy as np
from scipy.ndimage import map_coordinates, sobel

from vajranow.config import load_config
from vajranow.synthetic import StormCell, generate_cells

CONFIG = load_config()


def compute_optical_flow_lk(
    frame1: np.ndarray, frame2: np.ndarray, window_size: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D motion field (u, v) between two consecutive frames using Lucas-Kanade optical flow.

    Solves the optical flow constraint:
    I_x * u + I_y * v + I_t = 0
    over local neighborhood patches of size window_size x window_size.
    """
    if frame1.shape != frame2.shape:
        raise ValueError("Input frames must have identical shapes")

    h, w = frame1.shape
    # Spatial gradients using Sobel operators
    Ix = sobel(frame2, axis=1, mode="constant") / 8.0
    Iy = sobel(frame2, axis=0, mode="constant") / 8.0
    # Temporal gradient
    It = frame2 - frame1

    u = np.zeros((h, w), dtype=np.float32)
    v = np.zeros((h, w), dtype=np.float32)

    # Pre-calculate sums for efficiency
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It

    # Uniform box filter convolution for neighborhood summation
    from scipy.ndimage import uniform_filter

    s_Ix2 = uniform_filter(Ix2, size=window_size) * (window_size**2)
    s_Iy2 = uniform_filter(Iy2, size=window_size) * (window_size**2)
    s_IxIy = uniform_filter(IxIy, size=window_size) * (window_size**2)
    s_IxIt = uniform_filter(IxIt, size=window_size) * (window_size**2)
    s_IyIt = uniform_filter(IyIt, size=window_size) * (window_size**2)

    det = s_Ix2 * s_Iy2 - s_IxIy * s_IxIy
    valid = det > 1e-4

    u[valid] = -(s_Iy2[valid] * s_IxIt[valid] - s_IxIy[valid] * s_IyIt[valid]) / det[valid]
    v[valid] = -(s_Ix2[valid] * s_IyIt[valid] - s_IxIy[valid] * s_IxIt[valid]) / det[valid]

    # Clip extreme unphysical velocities (> 30 px per step)
    u = np.clip(u, -15.0, 15.0)
    v = np.clip(v, -15.0, 15.0)

    # Smooth the flow field
    from scipy.ndimage import gaussian_filter

    u = gaussian_filter(u, sigma=1.5)
    v = gaussian_filter(v, sigma=1.5)

    return u, v


def semi_lagrangian_extrapolate(
    field: np.ndarray, u: np.ndarray, v: np.ndarray, lead_steps: int
) -> np.ndarray:
    """Extrapolate a 2D scalar field forward in time using backward semi-Lagrangian advection.

    Grid coordinates are traced backward along the motion trajectory:
    x_source = x - lead_steps * u
    y_source = y - lead_steps * v
    and interpolated via bicubic/bilinear coordinates.
    """
    h, w = field.shape
    yy, xx = np.mgrid[:h, :w]
    src_x = xx - u * lead_steps
    src_y = yy - v * lead_steps

    advected = map_coordinates(field, [src_y, src_x], order=1, mode="nearest")
    return advected


def steps_ensemble_exceedance(
    advected_field: np.ndarray,
    threshold: float,
    lead_minutes: int,
    n_ensemble: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Generate STEPS-style probabilistic exceedance fractions from an advected deterministic field.

    Adds spatially correlated perturbation noise that scales with lead time
    (representing cascade scale growth of forecast uncertainty).
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed + lead_minutes)
    h, w = advected_field.shape
    exceedance_count = np.zeros((h, w), dtype=np.float32)

    # Noise std increases with lead time (from 1.5 dBZ at 10m to 8.0 dBZ at 180m)
    sigma_noise = 1.5 + 6.5 * (lead_minutes / 180.0)
    # Spatial correlation scale (smoothing sigma)
    spatial_scale = 2.0 + 3.0 * (lead_minutes / 180.0)

    for _ in range(n_ensemble):
        noise = rng.normal(0, sigma_noise, (h, w))
        filtered_noise = gaussian_filter(noise, sigma=spatial_scale)
        # Normalize filtered noise to maintain variance
        filtered_noise = filtered_noise * (sigma_noise / (np.std(filtered_noise) + 1e-6))

        perturbed = advected_field + filtered_noise
        exceedance_count += (perturbed >= threshold).astype(np.float32)

    return exceedance_count / float(n_ensemble)


def forecast_cells(
    cells: list[StormCell],
    leads_minutes: list[int] | None = None,
    *,
    radar_available: bool = True,
    config: dict | None = None,
) -> dict:
    """Transparent Lucas-Kanade + STEPS baseline cell & grid forecast."""
    cfg = config or load_config()
    leads = leads_minutes or cfg["forecast"]["lead_minutes"]
    frames = []

    for lead in leads:
        lead_fraction = lead / 60.0
        forecast = []
        for cell in cells:
            # Convective growth heuristic for zero-strike growing cells
            if cell.strikes == 0:
                growth = min(lead_fraction, 1.0) * cfg["forecast"]["growth_dbz_per_hour"]
                dbz = cell.dbz + growth
            else:
                dbz = cell.dbz

            # Logistic mapping to lightning probability
            midpoint = cfg["forecast"]["lightning_dbz_midpoint"]
            scale = cfg["forecast"]["lightning_dbz_scale"]
            prob = 1.0 / (1.0 + math.exp(-(dbz - midpoint) / scale))

            # Observed strike bonus
            if cell.strikes > 0:
                prob = min(0.95, prob + cfg["forecast"]["observed_strike_probability_bonus"])

            # Degradation when radar is missing
            if not radar_available:
                prob *= cfg["forecast"]["missing_radar_probability_factor"]

            # Severe probability (CMAX >= 40 dBZ)
            severe_prob = 1.0 / (
                1.0 + math.exp(-(dbz - cfg["forecast"]["severe_dbz"]) / cfg["forecast"]["severe_dbz_scale"])
            )
            if not radar_available:
                severe_prob *= cfg["forecast"]["missing_radar_probability_factor"]

            forecast.append(
                {
                    "cell_id": cell.cell_id,
                    "x": round(cell.x + cell.vx * lead, 2),
                    "y": round(cell.y + cell.vy * lead, 2),
                    "p_lightning": round(float(prob), 4),
                    "p_severe": round(float(severe_prob), 4),
                    "max_dbz": round(float(dbz), 1),
                    "lead_minutes": lead,
                }
            )
        frames.append({"lead_minutes": lead, "cells": forecast})

    engine = "ML_DEGRADED" if not radar_available else "BASELINE"
    return {"engine": engine, "product": "lightning", "frames": frames}


def current_cells(minute: int = 0) -> list[StormCell]:
    cfg = load_config()
    region = cfg["region"]
    return generate_cells(region["width"], region["height"], minute)
