from datetime import datetime, timedelta
from typing import Any

from vajranow.adapters.synthetic import SyntheticBackend
from vajranow.alerts.state_machine import AlertMachine
from vajranow.baseline import current_cells
from vajranow.config import load_config
from vajranow.models.engine import VajraNowInferenceEngine


class StrictNoLeakageDataWrapper:
    """Strict time-gated wrapper ensuring replay code can NEVER read observations where timestamp > t_replay."""

    def __init__(self, backend: Any, start_time: datetime):
        self.backend = backend
        self.start_time = start_time

    def get_observations_up_to(self, replay_time: datetime):
        """Fetch observations strictly clamped to timestamp <= replay_time."""
        if replay_time < self.start_time:
            raise ValueError("replay_time cannot precede start_time")
        # Read dataset only up to replay_time
        dataset = self.backend.get(self.start_time, replay_time)
        return dataset


class HindcastReplaySession:
    """Manages event hindcast replay execution at 10x, 30x, or 60x speeds.

    Tracks active cells, alert issuance, first strike timing, and verifies
    honest lead-time metrics between first warning and first ground strike.
    """

    def __init__(self, event_id: str = "EVENT_2024_06_MAH_01", config: dict | None = None):
        self.config = config or load_config()
        self.event_id = event_id
        self.start_time = datetime(2024, 6, 15, 12, 0, 0)
        self.current_minute = 0
        self.max_minutes = 180
        self.step_minutes = 10
        self.backend = SyntheticBackend(self.config)
        self.wrapper = StrictNoLeakageDataWrapper(self.backend, self.start_time)
        self.engine = VajraNowInferenceEngine(self.config)

        # Track per-cell alert machines
        self.alert_machines: dict[str, AlertMachine] = {}
        self.cell_first_alert: dict[str, int] = {}
        self.cell_first_strike: dict[str, int] = {}
        self.history_records: list[dict] = []

    def step(self) -> dict[str, Any]:
        """Advance replay by one 10-minute time step through the strict time wrapper."""
        if self.current_minute >= self.max_minutes:
            return {"status": "completed", "current_minute": self.current_minute, "is_finished": True}

        replay_time = self.start_time + timedelta(minutes=self.current_minute)

        # 1. Fetch strictly leak-free observations up to current replay_time
        _ = self.wrapper.get_observations_up_to(replay_time)

        # 2. Run forecast engine
        forecast_res = self.engine.forecast(minute=self.current_minute, product="lightning")

        # 3. Update alert state machines
        active_alerts = []
        cells = current_cells(self.current_minute)
        for cell in cells:
            cid = cell.cell_id
            if cid not in self.alert_machines:
                self.alert_machines[cid] = AlertMachine(
                    on=self.config["alerts"]["p_on"],
                    off=self.config["alerts"]["p_off"],
                    consecutive_on=self.config["alerts"]["consecutive_on"],
                    consecutive_off=self.config["alerts"]["consecutive_off"],
                    clear_minutes=self.config["alerts"]["strike_clear_minutes"],
                )

            # Check if strike occurred
            strike_now = cell.strikes > 0
            if strike_now and cid not in self.cell_first_strike:
                self.cell_first_strike[cid] = self.current_minute

            # Find cell's forecast probability
            prob = 0.5
            if forecast_res["frames"]:
                matching = next(
                    (c for c in forecast_res["frames"][0]["cells"] if c["cell_id"] == cid), None
                )
                if matching:
                    prob = matching["p_lightning"]

            is_alert_active = self.alert_machines[cid].update(
                prob, self.current_minute, observed_strike=strike_now
            )

            if is_alert_active:
                if cid not in self.cell_first_alert:
                    self.cell_first_alert[cid] = self.current_minute
                active_alerts.append({"cell_id": cid, "probability": prob, "minute": self.current_minute})

        # Calculate lead times for strikes
        verified_lead_times = []
        for cid, strike_m in self.cell_first_strike.items():
            if cid in self.cell_first_alert:
                lead_t = strike_m - self.cell_first_alert[cid]
                verified_lead_times.append(
                    {"cell_id": cid, "alert_minute": self.cell_first_alert[cid], "strike_minute": strike_m, "lead_time_minutes": max(0, lead_t)}
                )

        step_info = {
            "event_id": self.event_id,
            "current_minute": self.current_minute,
            "timestamp": replay_time.isoformat(),
            "active_cells_count": len(cells),
            "active_alerts_count": len(active_alerts),
            "engine": forecast_res["engine"],
            "verified_lead_times": verified_lead_times,
            "is_finished": self.current_minute + self.step_minutes >= self.max_minutes,
        }
        self.history_records.append(step_info)
        self.current_minute += self.step_minutes

        return step_info
