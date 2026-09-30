import numpy as np
import pytest

from vajranow.adapters.grid import composite_radars, regrid_nearest


def test_regrid_nearest_returns_configured_shape_and_values():
    source = np.array([[1, 2], [3, 4]], dtype=np.float32)
    result = regrid_nearest(source, (4, 4))
    assert result.shape == (4, 4)
    assert set(np.unique(result)) == {1, 2, 3, 4}


def test_radar_composite_uses_max_and_keeps_availability_channels():
    first = np.array([[20, np.nan], [30, 10]], dtype=np.float32)
    second = np.array([[25, 35], [np.nan, 5]], dtype=np.float32)
    composite, masks = composite_radars([first, second])
    np.testing.assert_array_equal(composite, [[25, 35], [30, 10]])
    assert masks.shape == (2, 2, 2)
    assert masks[0, 0, 1] == 0


def test_composite_handles_a_missing_radar_and_rejects_all_missing():
    composite, masks = composite_radars([np.ones((2, 2)), None])
    assert composite.shape == (2, 2)
    assert masks[1].sum() == 0
    with pytest.raises(ValueError):
        composite_radars([None, None])
