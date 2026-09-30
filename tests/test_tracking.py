import numpy as np

from vajranow.tracking import detect_cells, link_cells


def test_detect_cells_summarizes_connected_threshold_regions():
    field = np.zeros((12, 12), dtype=np.float32)
    field[2:4, 3:6] = 40
    cells = detect_cells(field, 32)
    assert len(cells) == 1
    assert cells[0]["area_pixels"] == 6
    assert cells[0]["max_dbz"] == 40


def test_link_cells_applies_distance_gate():
    previous = [{"x": 2.0, "y": 2.0}, {"x": 30.0, "y": 30.0}]
    current = [{"x": 3.0, "y": 2.0}, {"x": 50.0, "y": 50.0}]
    links = link_cells(previous, current, max_distance=5)
    assert len(links) == 1
    assert links[0]["previous_index"] == 0
