from vajranow.baseline import forecast_cells
from vajranow.synthetic import generate_cells, reflectivity_field


def test_synthetic_field_has_detectable_cells():
    field = reflectivity_field()
    assert field.shape == (64, 64)
    assert field.max() >= 40
    assert (field >= 32).any()


def test_baseline_forecasts_motion_and_engine():
    cell = generate_cells()[0]
    result = forecast_cells([cell], [10, 30])
    assert result["engine"] == "BASELINE"
    first, later = result["frames"]
    assert later["cells"][0]["x"] > first["cells"][0]["x"]
    assert first["cells"][0]["p_lightning"] >= 0


def test_missing_radar_is_explicitly_degraded():
    result = forecast_cells(generate_cells(), [10], radar_available=False)
    assert result["engine"] == "ML_DEGRADED"
