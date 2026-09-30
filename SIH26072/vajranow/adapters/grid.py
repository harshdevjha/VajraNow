import numpy as np
from scipy.ndimage import zoom


def regrid_nearest(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D field onto the configured grid using nearest-neighbor sampling."""
    if field.ndim != 2 or len(shape) != 2 or min(shape) < 1:
        raise ValueError("field and target shape must be non-empty 2-D grids")
    resized = zoom(field, (shape[0] / field.shape[0], shape[1] / field.shape[1]), order=0)
    return resized[: shape[0], : shape[1]]


def composite_radars(radars: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    """Composite radar arrays with max reflectivity and a per-radar availability mask."""
    available = [radar for radar in radars if radar is not None]
    if not available:
        raise ValueError("at least one radar grid is required to define the output shape")
    shape = available[0].shape
    if any(radar is not None and radar.shape != shape for radar in radars):
        raise ValueError("all radar grids must have the same shape")
    masks = np.stack([np.isfinite(radar) if radar is not None else np.zeros(shape, dtype=bool) for radar in radars])
    values = np.stack([np.asarray(radar, dtype=np.float32) if radar is not None else np.full(shape, np.nan, dtype=np.float32) for radar in radars])
    with np.errstate(all="ignore"):
        composite = np.max(np.where(np.isfinite(values), values, -np.inf), axis=0)
    composite[~np.isfinite(composite)] = 0.0
    return composite, masks.astype(np.uint8)
