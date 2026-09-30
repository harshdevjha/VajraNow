from datetime import UTC, datetime, timedelta

from vajranow.adapters.synthetic import SyntheticBackend


def test_synthetic_source_returns_aligned_steps_and_availability_masks():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    data = SyntheticBackend().get(start, start + timedelta(minutes=20))
    assert data.sizes["time"] == 3
    assert data.sizes["radar"] == 2
    assert data["composite_dbz"].shape == (3, 64, 64)
    assert data["radar_availability"].shape == (2, 3, 64, 64)
    assert data.attrs["label"] == "prototype, not operational"


def test_synthetic_source_rejects_reversed_intervals():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        SyntheticBackend().get(start + timedelta(minutes=10), start)
    except ValueError as error:
        assert "t1" in str(error)
    else:
        raise AssertionError("reversed interval should be rejected")
