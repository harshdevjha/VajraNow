from datetime import UTC, datetime
from xml.etree import ElementTree as ET

from vajranow.alerts.explainer import AlertExplainer
from vajranow.cap.cap_generator import create_cap
from vajranow.replay.hindcast import HindcastReplaySession
from vajranow.risk.exposure import RiskCalculator


def test_risk_calculator_exposure_weighting():
    calculator = RiskCalculator()
    sample_cells = [
        {"coordinates": [73.85, 18.52], "probability": 0.85, "p_lightning": 0.85, "max_dbz": 45.0}
    ]
    geojson = calculator.calculate_block_risks(sample_cells)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0
    pune_block = next((f for f in geojson["features"] if f["id"] == "MH-PUN-01"), None)
    assert pune_block is not None
    assert pune_block["properties"]["max_risk_score"] > 0.0
    assert pune_block["properties"]["risk_tier"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_alert_explainer_modality_ablation():
    explainer = AlertExplainer()
    ablation = explainer.explain_grid_modality_ablation(
        p_full=0.82, p_no_radar=0.55, p_no_sat=0.68, p_no_glm=0.74, p_no_nwp=0.79
    )
    assert ablation["attribution_type"] == "marginal_drop_delta_P"
    assert len(ablation["ablation_results"]) == 4
    assert ablation["ablation_results"][0]["delta_p"] == round(0.82 - 0.55, 4)


def test_cap_generator_structure_and_schema():
    xml_str = create_cap(
        alert_id="TEST-001",
        headline="Severe Storm Warning",
        description="Storm cell near Pune",
        sent=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
        coordinates=[73.8, 18.5],
        radius_km=15.0,
    )
    root = ET.fromstring(xml_str)
    assert "alert" in root.tag
    info = root.find("{urn:oasis:names:tc:emergency:cap:1.2}info")
    assert info is not None
    area = info.find("{urn:oasis:names:tc:emergency:cap:1.2}area")
    assert area is not None
    polygon = area.find("{urn:oasis:names:tc:emergency:cap:1.2}polygon")
    assert polygon is not None
    assert len(polygon.text.split()) >= 8


def test_hindcast_replay_no_leakage_session():
    session = HindcastReplaySession()
    assert session.current_minute == 0

    step1 = session.step()
    assert step1["current_minute"] == 0
    assert step1["is_finished"] is False

    step2 = session.step()
    assert step2["current_minute"] == 10
    assert session.current_minute == 20
