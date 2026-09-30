import pytest

from vajranow.alerts import AlertMachine


def test_alert_requires_consecutive_updates_and_hysteresis():
    machine = AlertMachine(0.6, 0.3, consecutive_on=2, consecutive_off=2, clear_minutes=30)
    assert machine.update(0.8, 0) is False
    assert machine.update(0.8, 10) is True
    assert machine.update(0.2, 20) is True
    assert machine.update(0.2, 30) is False


def test_recent_strike_delays_all_clear():
    machine = AlertMachine(0.6, 0.3, consecutive_on=1, consecutive_off=1, clear_minutes=30)
    assert machine.update(0.9, 0, observed_strike=True) is True
    assert machine.update(0.1, 20) is True
    assert machine.update(0.1, 40) is False


def test_hysteresis_thresholds_are_validated():
    with pytest.raises(ValueError):
        AlertMachine(0.3, 0.6, 2, 3, 30)
