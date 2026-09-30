from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StormCell:
    cell_id: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    dbz: float
    age_minutes: int
    strikes: int


def generate_cells(width: int = 64, height: int = 64, minute: int = 0) -> list[StormCell]:
    """Return deterministic proxy cells; these are not observations or skill evidence."""
    cells = [
        StormCell("SYN-001", 18.0 + minute * 0.10, 22.0 + minute * 0.04, 0.10, 0.04, 4.5, 43.0, 40 + minute, 2),
        StormCell("SYN-002", 43.0 - minute * 0.06, 37.0 - minute * 0.02, -0.06, -0.02, 3.4, 35.0, 20 + minute, 0),
        StormCell("SYN-003", 31.0 + minute * 0.02, 12.0 + minute * 0.08, 0.02, 0.08, 2.8, 30.0 + min(minute * 0.12, 8), minute, 0),
    ]
    return [cell for cell in cells if 0 <= cell.x < width and 0 <= cell.y < height]


def reflectivity_field(width: int = 64, height: int = 64, minute: int = 0) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    for cell in generate_cells(width, height, minute):
        distance2 = (xx - cell.x) ** 2 + (yy - cell.y) ** 2
        footprint = distance2 <= cell.radius**2
        profile = cell.dbz - np.sqrt(distance2) * 1.5
        field[footprint] = np.maximum(field[footprint], profile[footprint])
    return field
