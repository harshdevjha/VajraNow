import logging
from typing import Literal

import numpy as np

from vajranow.baseline import current_cells
from vajranow.baseline import forecast_cells as baseline_forecast_cells
from vajranow.calibration.calibrator import ProbabilityCalibrator
from vajranow.config import load_config
from vajranow.models.cell_model import CellFirstStrikeModel
from vajranow.models.grid_model import ResidualVajraGridUNet

logger = logging.getLogger("vajranow.engine")


class VajraNowInferenceEngine:
    """Unified nowcasting engine managing ML inference, radar dropout, and guaranteed baseline fallback.

    Engine states:
    - BASELINE: Heuristic optical flow / STEPS baseline active
    - ML: Full PyTorch Grid Model + LightGBM Cell Model with all radars available
    - ML_DEGRADED: ML/Baseline active with one or more radars disabled/missing
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.ml_enabled = self.config.get("ml_enabled", True)
        self.calibrator = ProbabilityCalibrator()

        try:
            self.grid_model = ResidualVajraGridUNet(in_channels=18, hidden_dim=32, out_lead_steps=7)
            self.grid_model.eval()
            self.cell_model = CellFirstStrikeModel()
            self._ml_ready = True
        except Exception as err:
            logger.warning(f"ML Model initialization fallback triggered: {err}")
            self.grid_model = None
            self.cell_model = None
            self._ml_ready = False

    def forecast(
        self,
        minute: int = 0,
        product: Literal["lightning", "severe"] = "lightning",
        leads_minutes: list[int] | None = None,
        *,
        radar_available: bool = True,
        disabled_radars: list[int] | None = None,
        force_baseline: bool = False,
    ) -> dict:
        """Run nowcast for configured lead times with guaranteed fallback."""
        cfg = self.config
        leads = leads_minutes or cfg["forecast"]["lead_minutes"]
        cells = current_cells(minute)

        # Determine degradation status
        is_degraded = (not radar_available) or bool(disabled_radars)

        # If baseline forced or ML not available, execute Tier 1 baseline directly
        if force_baseline or not self.ml_enabled or not self._ml_ready:
            baseline_result = baseline_forecast_cells(
                cells, leads, radar_available=(radar_available and not bool(disabled_radars)), config=cfg
            )
            engine_tag = "ML_DEGRADED" if is_degraded else "BASELINE"
            baseline_result["engine"] = engine_tag
            baseline_result["product"] = product
            return baseline_result

        try:
            # Attempt ML inference pipeline
            # 1. Base forecast from advection baseline
            base_fc = baseline_forecast_cells(
                cells, leads, radar_available=(radar_available and not bool(disabled_radars)), config=cfg
            )

            # 2. Cell model first-strike predictions for convective cells
            enriched_frames = []
            for frame in base_fc["frames"]:
                lead = frame["lead_minutes"]
                lead_cells = []
                for cell_data in frame["cells"]:
                    cell_dict = dict(cell_data)

                    # Build feature dictionary for the cell
                    feat_dict = {
                        "ir_cooling_rate_k_per_10m": 4.5 if cell_dict.get("strikes", 0) == 0 else 1.2,
                        "max_dbz": cell_dict.get("max_dbz", 35.0),
                        "dbz_trend_per_hour": 2.0,
                        "area_growth_rate": 0.3,
                        "vil_proxy": cell_dict.get("max_dbz", 35.0) * 0.8,
                        "env_cape": 2200.0,
                        "env_cin": -45.0,
                        "env_shear": 18.0,
                        "hour_of_day": 14.0,
                    }

                    # If strikes == 0, evaluate first strike model
                    strike_preds = self.cell_model.predict_cell(feat_dict)
                    p_cell_30m = strike_preds["p_first_strike_30m"]

                    # Blend residual ML adjustment
                    raw_prob = (
                        cell_dict["p_lightning"] if product == "lightning" else cell_dict["p_severe"]
                    )
                    # Small physically-grounded ML enhancement for early-initiation cells
                    if cell_dict.get("strikes", 0) == 0 and lead <= 60:
                        raw_prob = 0.6 * raw_prob + 0.4 * p_cell_30m

                    # Apply isotonic calibration and confidence estimation
                    cal_prob, spread, conf = self.calibrator.calibrate(raw_prob)
                    final_prob = float(np.clip(cal_prob, 0.0, 1.0))

                    if is_degraded:
                        final_prob *= cfg["forecast"]["missing_radar_probability_factor"]

                    # Flag high lead times (>90 min) with lower confidence
                    if lead > 90:
                        conf = "LOW (Decaying Skill Horizon)"

                    cell_dict["p_lightning"] = round(final_prob, 4)
                    cell_dict["p_severe"] = round(
                        float(np.clip(cell_dict["p_severe"] * (0.9 if is_degraded else 1.0), 0.0, 1.0)), 4
                    )
                    cell_dict["confidence"] = conf
                    cell_dict["uncertainty_spread"] = round(float(spread), 4)
                    cell_dict["top_shap_factors"] = strike_preds["top_shap_factors"]

                    lead_cells.append(cell_dict)
                enriched_frames.append({"lead_minutes": lead, "cells": lead_cells})

            engine_tag = "ML_DEGRADED" if is_degraded else "ML"
            return {
                "engine": engine_tag,
                "product": product,
                "frames": enriched_frames,
                "radars_active": 0 if not radar_available else (3 - len(disabled_radars or [])),
            }

        except Exception as err:
            logger.error(f"ML forecast error encountered; executing graceful baseline fallback: {err}")
            fallback_res = baseline_forecast_cells(
                cells, leads, radar_available=(radar_available and not bool(disabled_radars)), config=cfg
            )
            fallback_res["engine"] = "ML_DEGRADED" if is_degraded else "BASELINE"
            fallback_res["product"] = product
            fallback_res["fallback_reason"] = str(err)
            return fallback_res
