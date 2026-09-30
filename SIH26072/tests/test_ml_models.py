import numpy as np
import torch

from vajranow.calibration.calibrator import ProbabilityCalibrator
from vajranow.models.cell_model import CellFirstStrikeModel
from vajranow.models.engine import VajraNowInferenceEngine
from vajranow.models.grid_model import FocalLoss, ResidualVajraGridUNet


def test_residual_unet_zero_initialization():
    """Verify that before any training updates, final residual perturbation is zero."""
    model = ResidualVajraGridUNet(in_channels=18, hidden_dim=16, out_lead_steps=7)
    model.eval()

    x = torch.randn(1, 18, 64, 64)
    baseline_lightning = torch.zeros(1, 7, 64, 64)
    baseline_severe = torch.zeros(1, 7, 64, 64)

    p_lightning, p_severe = model(x, baseline_lightning, baseline_severe)

    # Sigmoid(0) == 0.5
    assert torch.allclose(p_lightning, torch.full_like(p_lightning, 0.5), atol=1e-5)
    assert torch.allclose(p_severe, torch.full_like(p_severe, 0.5), atol=1e-5)


def test_focal_loss_forward():
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.randn(4, 7, 32, 32)
    targets = (torch.rand(4, 7, 32, 32) > 0.8).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0


def test_cell_first_strike_model_prediction_and_shap():
    cell_model = CellFirstStrikeModel()
    features = {
        "ir_cooling_rate_k_per_10m": 6.5,
        "max_dbz": 44.0,
        "dbz_trend_per_hour": 5.0,
        "area_growth_rate": 0.4,
        "vil_proxy": 35.0,
        "env_cape": 2600.0,
        "env_cin": -30.0,
        "env_shear": 22.0,
        "hour_of_day": 15.0,
    }
    result = cell_model.predict_cell(features)
    assert 0.0 <= result["p_first_strike_30m"] <= 1.0
    assert 0.0 <= result["p_first_strike_60m"] <= 1.0
    assert len(result["top_shap_factors"]) > 0
    assert "shap_importance" in result["top_shap_factors"][0]


def test_probability_calibrator():
    calibrator = ProbabilityCalibrator()
    uncalibrated = np.array([0.1, 0.4, 0.7, 0.9])
    mean_p, spread, conf = calibrator.calibrate(uncalibrated)

    assert len(mean_p) == 4
    assert np.all(mean_p >= 0.0) and np.all(mean_p <= 1.0)
    assert np.all(spread >= 0.0)
    assert conf in {"HIGH", "MODERATE", "LOW"}


def test_inference_engine_fallback():
    engine = VajraNowInferenceEngine()
    fc = engine.forecast(minute=0, product="lightning", leads_minutes=[10, 30])
    assert fc["engine"] in {"ML", "BASELINE"}
    assert len(fc["frames"]) == 2
    assert "cells" in fc["frames"][0]
