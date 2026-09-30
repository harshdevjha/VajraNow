# VajraNow Architecture & Engineering Specification

## 1. System Overview
**VajraNow** is an AI/ML nowcasting platform designed for Problem Statement 26072: Multi-radar, satellite, lightning, and NWP model data fusion for thunderstorm and lightning nowcasting.

```mermaid
graph TD
    subgraph Data Strategy & Ingest Layer
        R[IMD DWR Radars] --> AD[Adapter Pipeline]
        S[MOSDAC INSAT-3D/3DR] --> AD
        L[IITM / Damini Lightning] --> AD
        M[GFS / ERA5 NWP Envs] --> AD
        SEVIR[SEVIR Proxy Backend] --> AD
        SYN[Synthetic Convective Generator] --> AD
    end

    subgraph Core Processing Pipeline
        AD --> GRID[Grid Regrid & Composite]
        GRID --> TRK[tobac/Hungarian Cell Tracker]
        GRID --> LK[Lucas-Kanade Optical Flow Baseline]
        LK --> EXT[Semi-Lagrangian Advection + STEPS]
    end

    subgraph Tier 2: AI/ML Inference & Calibration
        EXT --> ML_GRID[PyTorch Residual U-Net + ConvLSTM]
        TRK --> ML_CELL[LightGBM First-Strike Predictor]
        ML_GRID --> CALIB[Isotonic Calibrator + 3-Seed Ensemble]
        ML_CELL --> CALIB
        CALIB --> ENG{Engine Selector: BASELINE / ML / ML_DEGRADED}
    end

    subgraph Tier 3: Response, Exposure Risk & Explainability
        ENG --> SM[Alert Policy State Machine with Hysteresis]
        ENG --> WHY[SHAP TreeExplainer & Modality Ablation]
        ENG --> RISK[Exposure-Weighted Block Risk Aggregator]
        SM --> CAP[OASIS CAP 1.2 XML Alert Exporter]
        ENG --> REPLAY[Strict No-Leakage Hindcast Replayer]
        ENG --> SCORE[Held-Out Evaluation Scoreboard]
    end
```

## 2. Multi-Tier Architecture

### Tier 1: Working Skeleton & Baseline
- **Data Adapters**: Interchangeable `RealDataAdapter`, `SEVIRProxyAdapter`, and `SyntheticBackend` implementing `get(t0, t1) -> xarray.Dataset`.
- **Radar Compositing**: Max reflectivity composite ($Z = \max_i Z_i$) with per-radar availability channels to test radar dropout.
- **Cell Detection & Tracking**: Multi-threshold connected components (30-35 dBZ) + Hungarian matching (`scipy.optimize.linear_sum_assignment`) on advection-predicted centroids.
- **Optical Flow Baseline**: Deterministic Lucas-Kanade motion estimation + semi-Lagrangian backward advection to +180 min, combined with STEPS-style stochastic spatial noise perturbations for ensemble exceedance probabilities.

### Tier 2: AI/ML Nowcasting Engine
- **Grid Model (PyTorch)**: U-Net encoder-decoder with a ConvLSTM bottleneck.
  - Inputs: Past 6 radar frames + satellite IR bands (`ir069`, `ir107`) + 10-min IR cooling differences + GLM/network lightning density + NWP environment (CAPE, CIN, PWAT, Shear) + radar availability masks + baseline advected logits.
  - Residual formulation: $\text{logit}(P) = \text{logit}(P_{\text{advected, clipped}}) + f(X)$. Final layer zero-initialized for exact baseline equivalence at step 0.
  - Class-weighted Focal Loss to address extreme convective storm rarity.
  - Radar dropout during forward pass and training.
- **Cell Model (LightGBM)**: For cells with zero strikes so far, predicts $P(\text{first strike within 30 and 60 min})$ using IR cooling rate, max dBZ, area growth, and NWP instability.
- **Calibration & Uncertainty**: Isotonic regression fitted on a strictly held-out calibration split. 3-seed ensemble spread indicates epistemic confidence. Leads $>90\text{ min}$ are flagged as decaying confidence horizons.

### Tier 3: Response Layer & Explainability
- **Alert State Machine**: Hysteresis policy ($t_{\text{on}}=0.55$, $t_{\text{off}}=0.35$, $N_{\text{on}}=2$ consecutive cycles to alert, $N_{\text{off}}=3$ consecutive cycles below $t_{\text{off}}$ AND no strikes within 15 km for 30 min to issue All-Clear).
- **Explainability ("Why" Cards)**:
  - SHAP TreeExplainer for the LightGBM cell model.
  - Modality Ablation Attribution for the grid model ($\Delta P$ drop when Radar, Satellite, Lightning, or NWP are ablated).
- **Exposure-Weighted Risk**: $\text{Risk} = P(\text{hazard}) \times (w_{\text{pop}} \cdot \text{norm}(\text{pop}) + w_{\text{land}} \cdot \text{land\_mult})$. Aggregated into GeoJSON administrative block polygons.
- **OASIS CAP 1.2 XML**: Standardized emergency alerting XML format.
- **Hindcast Replay**: Strict time-gated wrapper ensuring no future data leakage ($\text{timestamp} \le t_{\text{replay}}$) with live step-by-step playback.
- **Honest Scoreboard**: Evaluator measuring CSI, POD, FAR, Bias, FSS @ 15 km, Brier Score, BSS, and Lead Time to First Strike on held-out events.
