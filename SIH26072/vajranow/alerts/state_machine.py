from dataclasses import dataclass


@dataclass
class AlertState:
    active: bool = False
    above_count: int = 0
    below_count: int = 0
    last_strike_minute: int | None = None
    activation_minute: int | None = None
    current_probability: float = 0.0


class AlertMachine:
    """Deterministic alert state machine with hysteresis and strike-proximity cooldown.

    Policy:
    - Activate: p >= t_on for consecutive_on consecutive update cycles.
    - Hysteresis: t_off < t_on.
    - All-Clear: p <= t_off for consecutive_off consecutive cycles AND
      no observed lightning strike within radius R for clear_minutes.
    """

    def __init__(
        self,
        on: float = 0.55,
        off: float = 0.35,
        consecutive_on: int = 2,
        consecutive_off: int = 3,
        clear_minutes: int = 30,
        radius_km: float = 15.0,
    ):
        if not 0 <= off < on <= 1:
            raise ValueError("alert thresholds must satisfy 0 <= off < on <= 1")
        self.on = on
        self.off = off
        self.consecutive_on = consecutive_on
        self.consecutive_off = consecutive_off
        self.clear_minutes = clear_minutes
        self.radius_km = radius_km
        self.state = AlertState()

    def update(self, probability: float, minute: int, observed_strike: bool = False) -> bool:
        state = self.state
        state.current_probability = probability
        if observed_strike:
            state.last_strike_minute = minute

        state.above_count = state.above_count + 1 if probability >= self.on else 0
        state.below_count = state.below_count + 1 if probability <= self.off else 0

        # Activation condition
        if not state.active and state.above_count >= self.consecutive_on:
            state.active = True
            state.activation_minute = minute

        # Deactivation condition
        recent_strike = (
            state.last_strike_minute is not None
            and (minute - state.last_strike_minute) < self.clear_minutes
        )
        if state.active and state.below_count >= self.consecutive_off and not recent_strike:
            state.active = False
            state.activation_minute = None

        return state.active
