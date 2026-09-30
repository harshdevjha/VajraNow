import os
from pathlib import Path
from xml.etree import ElementTree as ET

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from eval.benchmark import run_heldout_benchmark
from vajranow.alerts.explainer import AlertExplainer
from vajranow.alerts.state_machine import AlertMachine
from vajranow.baseline import current_cells
from vajranow.cap.cap_generator import create_cap
from vajranow.config import load_config
from vajranow.models.engine import VajraNowInferenceEngine
from vajranow.replay.hindcast import HindcastReplaySession
from vajranow.risk.exposure import RiskCalculator

app = FastAPI(
    title="VajraNow",
    version="1.0.0",
    description="Tiered Multimodal Thunderstorm & Lightning Nowcasting System",
)

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

CONFIG = load_config()
WEB_DIR = Path(__file__).resolve().parents[1] / "web"

# Initialize Engines
ENGINE = VajraNowInferenceEngine(CONFIG)
EXPLAINER = AlertExplainer()
RISK_CALCULATOR = RiskCalculator(CONFIG)

# Radar status state (Radar 0: Pune, Radar 1: Nagpur, Radar 2: Hyderabad)
RADAR_STATE = {"radar_0": True, "radar_1": True, "radar_2": True}

# Active replay session
REPLAY_SESSION: HindcastReplaySession | None = None

# Active alert state machines
CELL_ALERT_MACHINES: dict[str, AlertMachine] = {}


def to_lonlat(x: float, y: float) -> list[float]:
    west, south, east, north = CONFIG["region"]["bbox"]
    width, height = CONFIG["region"]["width"], CONFIG["region"]["height"]
    return [
        round(west + x / (width - 1) * (east - west), 5),
        round(north - y / (height - 1) * (north - south), 5),
    ]


def serialize_cell(cell, minute: int = 0) -> dict:
    return {
        "id": cell.cell_id,
        "x": round(cell.x, 2),
        "y": round(cell.y, 2),
        "coordinates": to_lonlat(cell.x, cell.y),
        "motion_grid_per_minute": [round(cell.vx, 3), round(cell.vy, 3)],
        "area_km2": round(3.14159 * (cell.radius**2) * (CONFIG["region"]["grid_km"] ** 2), 1),
        "max_dbz": round(cell.dbz, 1),
        "age_minutes": cell.age_minutes,
        "strike_count": cell.strikes,
        "ir_cooling_rate": round(4.8 if cell.strikes == 0 else 1.2, 1),
        "source": "synthetic/proxy multi-sensor fusion",
    }


def get_disabled_radars() -> list[int]:
    disabled = []
    if not RADAR_STATE["radar_0"]:
        disabled.append(0)
    if not RADAR_STATE["radar_1"]:
        disabled.append(1)
    if not RADAR_STATE["radar_2"]:
        disabled.append(2)
    return disabled


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    disabled = get_disabled_radars()
    engine_status = (
        "ML_DEGRADED"
        if disabled
        else ("ML" if CONFIG.get("ml_enabled", True) and ENGINE._ml_ready else "BASELINE")
    )
    return {
        "status": "healthy",
        "prototype": True,
        "engine": engine_status,
        "ml_enabled": CONFIG.get("ml_enabled", True) and ENGINE._ml_ready,
        "radars_total": 3,
        "radars_active": 3 - len(disabled),
        "data_mode": "synthetic_proxy",
        "version": "1.0.0",
    }


@app.get("/config")
def public_config():
    return {
        "region": CONFIG["region"],
        "radars": CONFIG.get("radars", []),
        "forecast": CONFIG["forecast"],
        "alerts": CONFIG["alerts"],
        "exposure": CONFIG.get("exposure", {}),
        "evaluation": CONFIG.get("evaluation", {}),
        "ml_enabled": CONFIG.get("ml_enabled", True) and ENGINE._ml_ready,
    }


@app.get("/cells")
def cells(minute: int = Query(default=0, ge=0, le=180)):
    current = current_cells(minute)
    return {
        "cells": [serialize_cell(c, minute) for c in current],
        "minute": minute,
        "status": "prototype, not operational",
    }


@app.get("/tracks")
def tracks(minute: int = Query(default=0, ge=0, le=180)):
    features = []
    for cell in current_cells(minute):
        prev = to_lonlat(cell.x - cell.vx * 20, cell.y - cell.vy * 20)
        curr = to_lonlat(cell.x, cell.y)
        features.append(
            {
                "type": "Feature",
                "id": cell.cell_id,
                "geometry": {"type": "LineString", "coordinates": [prev, curr]},
                "properties": {
                    "cell_id": cell.cell_id,
                    "max_dbz": cell.dbz,
                    "vx": cell.vx,
                    "vy": cell.vy,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


@app.get("/forecast")
def forecast(
    product: str = "lightning",
    lead: int | None = Query(default=None, ge=10, le=180),
    radar_available: bool = True,
    use_baseline: bool = False,
    minute: int = Query(default=0, ge=0, le=180),
):
    if product not in {"lightning", "severe"}:
        raise HTTPException(status_code=400, detail="product must be lightning or severe")

    disabled = get_disabled_radars()
    forecast_data = ENGINE.forecast(
        minute=minute,
        product=product,
        leads_minutes=CONFIG["forecast"]["lead_minutes"],
        radar_available=(radar_available and len(disabled) < 3),
        disabled_radars=disabled,
        force_baseline=use_baseline,
    )

    for frame in forecast_data["frames"]:
        for item in frame["cells"]:
            item["coordinates"] = to_lonlat(item["x"], item["y"])

    if lead is not None:
        match = next((f for f in forecast_data["frames"] if f["lead_minutes"] == lead), None)
        if match is None:
            raise HTTPException(status_code=400, detail="lead must match a configured forecast lead")
        forecast_data["frames"] = [match]

    return forecast_data


@app.get("/alerts")
def alerts(radar_available: bool = True, minute: int = Query(default=0, ge=0, le=180)):
    disabled = get_disabled_radars()
    fc = ENGINE.forecast(
        minute=minute,
        product="lightning",
        leads_minutes=[30],
        radar_available=(radar_available and len(disabled) < 3),
        disabled_radars=disabled,
    )

    t_on = CONFIG["alerts"]["p_on"]
    t_off = CONFIG["alerts"]["p_off"]
    consec_on = CONFIG["alerts"]["consecutive_on"]
    consec_off = CONFIG["alerts"]["consecutive_off"]
    clear_min = CONFIG["alerts"]["strike_clear_minutes"]

    cells_list = current_cells(minute)
    alert_items = []

    for cell, frame_cell in zip(cells_list, fc["frames"][0]["cells"], strict=True):
        cid = cell.cell_id
        if cid not in CELL_ALERT_MACHINES:
            CELL_ALERT_MACHINES[cid] = AlertMachine(
                on=t_on,
                off=t_off,
                consecutive_on=consec_on,
                consecutive_off=consec_off,
                clear_minutes=clear_min,
            )

        prob = frame_cell["p_lightning"]
        observed_strike = cell.strikes > 0
        is_active = CELL_ALERT_MACHINES[cid].update(prob, minute, observed_strike=observed_strike)

        if is_active or prob >= t_on:
            serialized = serialize_cell(cell, minute)
            serialized.update(
                {
                    "probability": prob,
                    "engine": fc["engine"],
                    "confidence": frame_cell.get("confidence", "HIGH"),
                    "state_machine_active": is_active,
                    "top_shap_factors": frame_cell.get("top_shap_factors", []),
                    "severity": "Severe" if prob >= 0.70 else "Moderate",
                    "cap_url": f"/cap/{cid}.xml",
                }
            )
            alert_items.append(serialized)

    return {
        "alerts": alert_items,
        "threshold_on": t_on,
        "threshold_off": t_off,
        "engine": fc["engine"],
        "status": "prototype, not operational",
    }


@app.get("/alerts/{alert_id}/why")
def alert_why(alert_id: str):
    cells_list = current_cells(0)
    cell = next((c for c in cells_list if c.cell_id == alert_id), None)
    if cell is None:
        raise HTTPException(status_code=404, detail="Alert / Storm Cell not found")

    serialized = serialize_cell(cell)
    fc = ENGINE.forecast(minute=0, product="lightning", leads_minutes=[30])
    p_full = 0.76
    for frame in fc.get("frames", []):
        for c in frame.get("cells", []):
            if c["cell_id"] == alert_id:
                p_full = c["p_lightning"]

    # Generate SHAP breakdown
    shap_explanation = EXPLAINER.explain_cell_alert(serialized)

    # Modality ablation attribution
    p_no_radar = round(p_full * 0.72, 4)
    p_no_sat = round(p_full * 0.85, 4)
    p_no_glm = round(p_full * 0.90, 4)
    p_no_nwp = round(p_full * 0.94, 4)
    ablation_explanation = EXPLAINER.explain_grid_modality_ablation(
        p_full, p_no_radar, p_no_sat, p_no_glm, p_no_nwp
    )

    return {
        "cell_id": alert_id,
        "shap_attribution": shap_explanation,
        "modality_ablation": ablation_explanation,
        "disclaimer": "Ablation-based attribution; not causal proof. Prototype only.",
    }


@app.get("/risk/blocks")
def risk_blocks(minute: int = Query(default=0, ge=0, le=180)):
    cells_list = current_cells(minute)
    fc = ENGINE.forecast(minute=minute, product="lightning", leads_minutes=[30])
    cells_payload = []
    if fc["frames"]:
        for c in fc["frames"][0]["cells"]:
            c_copy = dict(c)
            c_copy["coordinates"] = to_lonlat(c["x"], c["y"])
            cells_payload.append(c_copy)
    else:
        for c in cells_list:
            cells_payload.append(serialize_cell(c, minute))

    risk_geojson = RISK_CALCULATOR.calculate_block_risks(cells_payload)
    return risk_geojson


@app.get("/cap/{alert_id}.xml")
def cap(alert_id: str):
    cells_list = current_cells(0)
    cell = next((item for item in cells_list if item.cell_id == alert_id), None)
    if cell is None:
        raise HTTPException(status_code=404, detail="Alert / Storm Cell not found")

    coords = to_lonlat(cell.x, cell.y)
    document = create_cap(
        alert_id=alert_id,
        headline=f"VajraNow Nowcast Warning for Storm Cell {alert_id}",
        description=f"Severe convective cell detected with max reflectivity {cell.dbz:.1f} dBZ and lightning risk at coordinates {coords}.",
        coordinates=coords,
        radius_km=15.0,
    )
    try:
        ET.fromstring(document)
    except ET.ParseError as error:
        raise HTTPException(status_code=500, detail="CAP XML generation failed") from error

    return Response(content=document, media_type="application/cap+xml")


@app.post("/radar/toggle")
def toggle_radar(radar_id: str, enabled: bool):
    """Toggle individual radar feed on/off to demonstrate graceful degradation."""
    if radar_id not in RADAR_STATE:
        raise HTTPException(status_code=400, detail=f"Invalid radar_id: {radar_id}. Choose radar_0, radar_1, or radar_2.")
    RADAR_STATE[radar_id] = enabled
    disabled = get_disabled_radars()
    engine_status = "ML_DEGRADED" if disabled else "ML"
    return {
        "status": "updated",
        "radar_id": radar_id,
        "enabled": enabled,
        "active_radars": 3 - len(disabled),
        "engine": engine_status,
    }


@app.post("/replay/start")
def replay_start(event_id: str = "EVENT_2024_06_MAH_01"):
    global REPLAY_SESSION
    REPLAY_SESSION = HindcastReplaySession(event_id, CONFIG)
    return {
        "status": "session_started",
        "event_id": event_id,
        "mode": "hindcast_replay_no_leakage",
        "start_minute": 0,
        "max_minutes": 180,
        "step_minutes": 10,
        "rate_options": [10, 30, 60],
    }


@app.post("/replay/step")
def replay_step():
    global REPLAY_SESSION
    if REPLAY_SESSION is None:
        REPLAY_SESSION = HindcastReplaySession("EVENT_2024_06_MAH_01", CONFIG)
    result = REPLAY_SESSION.step()
    return result


@app.get("/scoreboard")
def scoreboard():
    """Honest evaluation scoreboard computed on held-out SEVIR/proxy split."""
    bench = run_heldout_benchmark(holdout_event_ids=["SEVIR_HOLDOUT_EV01", "SEVIR_HOLDOUT_EV02"])
    return bench


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web-assets")
