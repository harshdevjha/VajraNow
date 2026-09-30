from fastapi.testclient import TestClient

from vajranow.api import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["prototype"] is True
    assert data["engine"] in {"ML", "BASELINE", "ML_DEGRADED"}


def test_forecast_reports_ml_and_graceful_degradation():
    ml_res = client.get("/forecast", params={"lead": 30})
    baseline_res = client.get("/forecast", params={"lead": 30, "use_baseline": "true"})
    degraded = client.get("/forecast", params={"lead": 30, "radar_available": "false"})

    assert ml_res.status_code == 200
    assert ml_res.json()["engine"] in {"ML", "BASELINE"}
    assert baseline_res.json()["engine"] == "BASELINE"
    assert degraded.json()["engine"] == "ML_DEGRADED"
    assert degraded.json()["frames"][0]["lead_minutes"] == 30


def test_cells_and_tracks_endpoints():
    cells_resp = client.get("/cells", params={"minute": 10})
    tracks_resp = client.get("/tracks", params={"minute": 10})

    assert cells_resp.status_code == 200
    assert len(cells_resp.json()["cells"]) > 0
    assert tracks_resp.status_code == 200
    assert tracks_resp.json()["type"] == "FeatureCollection"


def test_alerts_and_why_explanation():
    alerts_resp = client.get("/alerts")
    assert alerts_resp.status_code == 200
    assert "alerts" in alerts_resp.json()

    # Why endpoint
    why_resp = client.get("/alerts/SYN-001/why")
    assert why_resp.status_code == 200
    data = why_resp.json()
    assert "shap_attribution" in data
    assert "modality_ablation" in data
    assert "Ablation-based attribution" in data["disclaimer"]


def test_risk_blocks_endpoint():
    risk_resp = client.get("/risk/blocks")
    assert risk_resp.status_code == 200
    data = risk_resp.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0
    assert "max_risk_score" in data["features"][0]["properties"]


def test_cap_endpoint_returns_valid_xml():
    response = client.get("/cap/SYN-001.xml")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/cap+xml")
    assert "urn:oasis:names:tc:emergency:cap:1.2" in response.text
    assert "<identifier>SYN-001</identifier>" in response.text


def test_radar_toggle_and_degradation():
    toggle_resp = client.post("/radar/toggle", params={"radar_id": "radar_0", "enabled": False})
    assert toggle_resp.status_code == 200
    data = toggle_resp.json()
    assert data["engine"] == "ML_DEGRADED"
    assert data["active_radars"] == 2

    # Restore radar
    restore_resp = client.post("/radar/toggle", params={"radar_id": "radar_0", "enabled": True})
    assert restore_resp.status_code == 200
    assert restore_resp.json()["active_radars"] == 3


def test_replay_session_start_and_step():
    start_resp = client.post("/replay/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["mode"] == "hindcast_replay_no_leakage"

    step_resp = client.post("/replay/step")
    assert step_resp.status_code == 200
    assert "current_minute" in step_resp.json()


def test_scoreboard_returns_evaluated_heldout_metrics():
    resp = client.get("/scoreboard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "evaluated_held_out"
    assert "VajraNow_ML" in data["models"]
    assert "LucasKanade_OpticalFlow" in data["models"]
    assert "Persistence" in data["models"]
    assert "CSI" in data["models"]["VajraNow_ML"]
    assert "FSS_15km" in data["models"]["VajraNow_ML"]
