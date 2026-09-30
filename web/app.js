// VajraNow Front-End Application Controller
const API_BASE = window.location.port === "8000" ? "" : "/api";

let config = null;
let map = null;
let currentLead = 30;
let currentProduct = "lightning";
let currentEngineMode = "ML"; // "ML" or "BASELINE"
let currentMinute = 0;
let selectedCellId = null;
let isReplayRunning = false;
let replayTimer = null;
let replaySpeed = 10;
let radarStates = { radar_0: true, radar_1: true, radar_2: true };

// UTC Clock
function updateClock() {
  const now = new Date();
  const utcStr = now.toISOString().substring(11, 19) + " UTC";
  const clockEl = document.querySelector("#clock");
  if (clockEl) clockEl.textContent = utcStr;
}
setInterval(updateClock, 1000);
updateClock();

// Initialize App
async function initApp() {
  try {
    const cfgResp = await fetch(`${API_BASE}/config`);
    config = await cfgResp.json();

    setupUIEvents();
    initMap();
    refreshData();
    // Periodic refresh every 10s if not in replay mode
    setInterval(() => {
      if (!isReplayRunning) refreshData();
    }, 10000);
  } catch (err) {
    console.error("Failed to initialize VajraNow console:", err);
  }
}

function setupUIEvents() {
  // Engine toggle
  document.querySelector("#mode-ml")?.addEventListener("click", () => setEngineMode("ML"));
  document.querySelector("#mode-baseline")?.addEventListener("click", () => setEngineMode("BASELINE"));

  // Product toggle
  document.querySelector("#prod-lightning")?.addEventListener("click", () => setProduct("lightning"));
  document.querySelector("#prod-severe")?.addEventListener("click", () => setProduct("severe"));

  // Lead slider
  const slider = document.querySelector("#lead-slider");
  slider?.addEventListener("input", (e) => {
    currentLead = parseInt(e.target.value, 10);
    document.querySelector("#lead-value").textContent = `T+${currentLead} min`;
    updateConfidenceBadge(currentLead);
    refreshData();
  });

  // Radar dropout toggles
  ["radar_0", "radar_1", "radar_2"].forEach((rId) => {
    const el = document.querySelector(`#toggle-${rId.replace("_", "-")}`);
    el?.addEventListener("change", async (e) => {
      radarStates[rId] = e.target.checked;
      await toggleRadar(rId, e.target.checked);
    });
  });

  // Scoreboard modal
  document.querySelector("#btn-scoreboard")?.addEventListener("click", showScoreboardModal);
  document.querySelector("#btn-close-scoreboard")?.addEventListener("click", hideScoreboardModal);
  document.querySelector("#scoreboard-modal")?.addEventListener("click", (e) => {
    if (e.target.id === "scoreboard-modal") hideScoreboardModal();
  });

  // Replay controls
  document.querySelector("#btn-replay-modal")?.addEventListener("click", toggleReplayMode);
  document.querySelector("#btn-replay-play")?.addEventListener("click", toggleReplayPlay);
  document.querySelector("#btn-replay-step")?.addEventListener("click", stepReplay);
  document.querySelector("#btn-replay-close")?.addEventListener("click", exitReplayMode);

  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".speed-btn").forEach((b) => b.classList.remove("active"));
      e.target.classList.add("active");
      replaySpeed = parseInt(e.target.dataset.speed, 10);
    });
  });
}

function updateConfidenceBadge(lead) {
  const badge = document.querySelector("#confidence-badge");
  const text = document.querySelector("#conf-text");
  if (!badge || !text) return;

  badge.className = "confidence-badge";
  if (lead <= 60) {
    badge.classList.add("conf-high");
    text.textContent = "HIGH CONFIDENCE (Physical Scaling Valid)";
  } else if (lead <= 90) {
    badge.classList.add("conf-moderate");
    text.textContent = "MODERATE CONFIDENCE (Convective Dispersion)";
  } else {
    badge.classList.add("conf-low");
    text.textContent = "LOW CONFIDENCE (Skill Horizon Decaying)";
  }
}

function setEngineMode(mode) {
  currentEngineMode = mode;
  document.querySelector("#mode-ml")?.classList.toggle("active", mode === "ML");
  document.querySelector("#mode-baseline")?.classList.toggle("active", mode === "BASELINE");
  refreshData();
}

function setProduct(prod) {
  currentProduct = prod;
  document.querySelector("#prod-lightning")?.classList.toggle("active", prod === "lightning");
  document.querySelector("#prod-severe")?.classList.toggle("active", prod === "severe");
  refreshData();
}

async function toggleRadar(radarId, enabled) {
  try {
    const res = await fetch(`${API_BASE}/radar/toggle?radar_id=${radarId}&enabled=${enabled}`, {
      method: "POST",
    });
    const data = await res.json();
    updateEngineBadge(data.engine);
    refreshData();
  } catch (err) {
    console.error("Failed to toggle radar:", err);
  }
}

function updateEngineBadge(engine) {
  const badge = document.querySelector("#engine-badge");
  const text = document.querySelector("#engine-text");
  if (!badge || !text) return;

  badge.className = "engine-badge";
  if (engine === "ML") {
    badge.classList.add("badge-ml");
    text.textContent = "ML (Residual U-Net + LGBM)";
  } else if (engine === "ML_DEGRADED") {
    badge.classList.add("badge-degraded");
    text.textContent = "ML DEGRADED (Radar Missing)";
  } else {
    badge.classList.add("badge-baseline");
    text.textContent = "BASELINE (LK Optical Flow)";
  }
}

// Map Initialization
function initMap() {
  const bbox = config?.region?.bbox || [73.0, 18.0, 79.0, 23.0];
  const center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2];

  try {
    map = new maplibregl.Map({
      container: "map",
      center: center,
      zoom: 6.0,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors · Synthetic Proxy Nowcasting",
          },
        },
        layers: [
          {
            id: "dark-base",
            type: "raster",
            source: "osm",
            paint: { "raster-saturation": -0.85, "raster-brightness-max": 0.4, "raster-contrast": 0.2 },
          },
        ],
      },
      maxBounds: [
        [bbox[0] - 2.5, bbox[1] - 2.0],
        [bbox[2] + 2.5, bbox[3] + 2.0],
      ],
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");

    map.on("load", () => {
      // Risk Blocks Layer
      map.addSource("risk-blocks", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "risk-blocks-fill",
        type: "fill",
        source: "risk-blocks",
        paint: {
          "fill-color": ["get", "risk_color"],
          "fill-opacity": 0.18,
        },
      });
      map.addLayer({
        id: "risk-blocks-border",
        type: "line",
        source: "risk-blocks",
        paint: {
          "line-color": ["get", "risk_color"],
          "line-width": 1.5,
          "line-opacity": 0.6,
        },
      });

      // Motion Tracks Layer
      map.addSource("tracks", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "tracks-line",
        type: "line",
        source: "tracks",
        paint: {
          "line-color": "#38bdf8",
          "line-width": 2,
          "line-dasharray": [3, 2],
          "line-opacity": 0.9,
        },
      });

      // Cell Halos (Reflectivity / Risk glow)
      map.addSource("cells", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "cell-halo",
        type: "circle",
        source: "cells",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "dbz"], 30, 16, 50, 36],
          "circle-color": [
            "interpolate",
            ["linear"],
            ["get", "dbz"],
            30, "#22c55e",
            38, "#eab308",
            45, "#f97316",
            55, "#dc2626",
          ],
          "circle-opacity": 0.35,
          "circle-blur": 0.6,
        },
      });

      // Cell Core Point
      map.addLayer({
        id: "cell-core",
        type: "circle",
        source: "cells",
        paint: {
          "circle-radius": 7,
          "circle-color": [
            "case",
            [">=", ["get", "dbz"], 40], "#ef4444",
            [">=", ["get", "dbz"], 35], "#f59e0b",
            "#10b981",
          ],
          "circle-stroke-width": 2,
          "circle-stroke-color": "#ffffff",
        },
      });

      map.on("click", "cell-core", (e) => {
        if (e.features && e.features[0]) {
          const props = e.features[0].properties;
          loadWhyCard(props.cell_id || props.id);
        }
      });

      map.on("mouseenter", "cell-core", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "cell-core", () => { map.getCanvas().style.cursor = ""; });
    });
  } catch (err) {
    console.warn("MapLibre initialization fallback to canvas:", err);
    initFallbackCanvas();
  }
}

function initFallbackCanvas() {
  const host = document.querySelector("#map");
  host.innerHTML = `<canvas id="fallback-canvas" style="width:100%;height:100%;background:#090d16;"></canvas>`;
}

// Data Fetch & Rendering
async function refreshData() {
  try {
    const isBaseline = currentEngineMode === "BASELINE";
    const [fcResp, cellsResp, tracksResp, alertsResp, riskResp] = await Promise.all([
      fetch(`${API_BASE}/forecast?product=${currentProduct}&lead=${currentLead}&minute=${currentMinute}&use_baseline=${isBaseline}`),
      fetch(`${API_BASE}/cells?minute=${currentMinute}`),
      fetch(`${API_BASE}/tracks?minute=${currentMinute}`),
      fetch(`${API_BASE}/alerts?minute=${currentMinute}`),
      fetch(`${API_BASE}/risk/blocks?minute=${currentMinute}`),
    ]);

    const fcData = await fcResp.json();
    const cellsData = await cellsResp.json();
    const tracksData = await tracksResp.json();
    const alertsData = await alertsResp.json();
    const riskData = await riskResp.json();

    updateEngineBadge(fcData.engine);
    renderMapData(fcData, tracksData, riskData);
    renderAlerts(alertsData);
    renderRiskSummary(riskData);

    // Update HUD
    document.querySelector("#map-active-cells").textContent = cellsData.cells.length;
    const activeRadars = (radarStates.radar_0 ? 1 : 0) + (radarStates.radar_1 ? 1 : 0) + (radarStates.radar_2 ? 1 : 0);
    document.querySelector("#map-radar-status").textContent = `${activeRadars}/3 ONLINE`;
    if (activeRadars < 3) {
      document.querySelector("#map-radar-status").className = "hud-val text-amber";
    } else {
      document.querySelector("#map-radar-status").className = "hud-val text-green";
    }

    if (selectedCellId) {
      loadWhyCard(selectedCellId);
    }
  } catch (err) {
    console.error("Data refresh failed:", err);
  }
}

function renderMapData(fcData, tracksData, riskData) {
  if (!map || !map.getSource) return;

  // Build GeoJSON features for forecast cells
  const cellFeatures = [];
  if (fcData.frames && fcData.frames[0]) {
    fcData.frames[0].cells.forEach((cell) => {
      cellFeatures.push({
        type: "Feature",
        id: cell.cell_id,
        geometry: { type: "Point", coordinates: cell.coordinates },
        properties: {
          cell_id: cell.cell_id,
          id: cell.cell_id,
          dbz: cell.max_dbz,
          p_lightning: cell.p_lightning,
          p_severe: cell.p_severe,
          confidence: cell.confidence,
        },
      });
    });
  }

  const cellsGeoJson = { type: "FeatureCollection", features: cellFeatures };

  if (map.getSource("cells")) map.getSource("cells").setData(cellsGeoJson);
  if (map.getSource("tracks")) map.getSource("tracks").setData(tracksData);
  if (map.getSource("risk-blocks")) map.getSource("risk-blocks").setData(riskData);
}

function renderAlerts(alertsData) {
  const listEl = document.querySelector("#alerts-list");
  const counter = document.querySelector("#alert-counter-badge");
  if (!listEl || !counter) return;

  const alerts = alertsData.alerts || [];
  counter.textContent = `${alerts.length} Active`;

  if (alerts.length === 0) {
    listEl.innerHTML = `<div class="empty-state">No convective cells currently breach threshold (${alertsData.threshold_on || 0.55}).</div>`;
    return;
  }

  listEl.innerHTML = alerts
    .map((alert) => {
      const pPercent = Math.round(alert.probability * 100);
      return `
      <div class="alert-item-card" onclick="loadWhyCard('${alert.id}')">
        <div class="alert-item-header">
          <span class="alert-cell-id">⚡ Cell ${alert.id}</span>
          <span class="alert-prob-tag">${pPercent}% Risk</span>
        </div>
        <div class="alert-metrics-row">
          <span>Max: <strong>${alert.max_dbz} dBZ</strong></span>
          <span>Strikes: <strong>${alert.strike_count}</strong></span>
          <span>Cooling: <strong>${alert.ir_cooling_rate} K/10m</strong></span>
        </div>
        <div class="alert-actions-row">
          <span class="mono-text">State: ${alert.state_machine_active ? "TRIGGERED (Hysteresis)" : "BREACH"}</span>
          <a class="cap-download-link" href="${alert.cap_url}" target="_blank" onclick="event.stopPropagation()">📥 CAP 1.2 XML</a>
        </div>
      </div>
    `;
    })
    .join("");
}

async function loadWhyCard(cellId) {
  selectedCellId = cellId;
  const contentEl = document.querySelector("#why-card-content");
  if (!contentEl) return;

  contentEl.innerHTML = `<div class="loading-spinner" style="font-size:11px;padding:8px 0;">Evaluating SHAP & modality ablations for ${cellId}...</div>`;

  try {
    const res = await fetch(`${API_BASE}/alerts/${cellId}/why`);
    if (!res.ok) throw new Error("Cell not found");
    const data = await res.json();

    const shapFactors = data.shap_attribution?.factors || [];
    const ablations = data.modality_ablation?.ablation_results || [];

    const shapRowsHtml = shapFactors
      .map((f) => {
        const isPos = f.contribution.startsWith("+");
        const fillClass = isPos ? "shap-fill-risk" : "shap-fill-calm";
        const widthPercent = Math.min(100, Math.abs(parseFloat(f.contribution)) * 300);
        return `
        <div class="shap-row">
          <div class="shap-meta">
            <span class="fname">${f.factor}</span>
            <span class="fval mono-text">${f.contribution} (${f.value})</span>
          </div>
          <div class="shap-progress-track">
            <div class="shap-progress-fill ${fillClass}" style="width: ${widthPercent}%;"></div>
          </div>
        </div>
      `;
      })
      .join("");

    const ablationRowsHtml = ablations
      .map((a) => {
        return `
        <tr>
          <td>${a.modality}</td>
          <td class="drop-impact mono-text">-${Math.round(a.delta_p * 100)}%</td>
          <td class="mono-text">${Math.round(a.retained_p * 100)}%</td>
        </tr>
      `;
      })
      .join("");

    contentEl.innerHTML = `
      <div class="why-container">
        <div class="why-header">
          <span class="why-title">⚡ Cell ${cellId} Attribution</span>
          <span class="why-sub">SHAP TreeExplainer</span>
        </div>
        <div class="shap-bars">
          ${shapRowsHtml}
        </div>
        
        <div class="why-header" style="margin-top:6px;">
          <span class="why-title" style="color:#f59e0b;">Modality Ablation Sensitivity</span>
          <span class="why-sub">ΔP drop on removal</span>
        </div>
        <table class="ablation-table">
          <thead>
            <tr><th>Modality Omitted</th><th>ΔP Drop</th><th>Retained P</th></tr>
          </thead>
          <tbody>
            ${ablationRowsHtml}
          </tbody>
        </table>
        <div class="why-sub" style="font-size:8px;color:#64748b;font-style:italic;">
          * Ablation-based attribution; sensitivity to input omission, not causal proof.
        </div>
      </div>
    `;
  } catch (err) {
    contentEl.innerHTML = `<div class="empty-state">Could not load explanation for ${cellId}.</div>`;
  }
}

function renderRiskSummary(riskData) {
  const container = document.querySelector("#risk-blocks-summary");
  if (!container || !riskData.features) return;

  container.innerHTML = riskData.features
    .slice(0, 4)
    .map((f) => {
      const p = f.properties;
      const tierClass = `tier-${p.risk_tier.toLowerCase()}`;
      return `
      <div class="risk-block-card">
        <div>
          <strong>${p.name}</strong>
          <div style="font-size:8.5px;color:#94a3b8;">Pop: ${p.population_density}/km² · ${p.land_cover}</div>
        </div>
        <span class="risk-tier-pill ${tierClass}">${p.risk_tier} (${Math.round(p.max_risk_score * 100)}%)</span>
      </div>
    `;
    })
    .join("");
}

// Replay Controller
function toggleReplayMode() {
  const toolbar = document.querySelector("#replay-toolbar");
  toolbar?.classList.toggle("hidden");
  if (!toolbar.classList.contains("hidden")) {
    startReplaySession();
  } else {
    exitReplayMode();
  }
}

async function startReplaySession() {
  try {
    await fetch(`${API_BASE}/replay/start`, { method: "POST" });
    currentMinute = 0;
    document.querySelector("#replay-time-label").textContent = `T+0 min (Replay)`;
    refreshData();
  } catch (err) {
    console.error("Replay start error:", err);
  }
}

function toggleReplayPlay() {
  const btn = document.querySelector("#btn-replay-play");
  if (isReplayRunning) {
    clearInterval(replayTimer);
    isReplayRunning = false;
    btn.textContent = "▶ Play";
  } else {
    isReplayRunning = true;
    btn.textContent = "⏸ Pause";
    const intervalMs = Math.max(200, 3000 / (replaySpeed / 10));
    replayTimer = setInterval(stepReplay, intervalMs);
  }
}

async function stepReplay() {
  try {
    const res = await fetch(`${API_BASE}/replay/step`, { method: "POST" });
    const data = await res.json();
    currentMinute = data.current_minute;
    document.querySelector("#replay-time-label").textContent = `T+${currentMinute} min (No Leakage)`;
    refreshData();

    if (data.is_finished) {
      clearInterval(replayTimer);
      isReplayRunning = false;
      document.querySelector("#btn-replay-play").textContent = "▶ Replay Done";
    }
  } catch (err) {
    console.error("Replay step error:", err);
  }
}

function exitReplayMode() {
  clearInterval(replayTimer);
  isReplayRunning = false;
  currentMinute = 0;
  document.querySelector("#replay-toolbar")?.classList.add("hidden");
  refreshData();
}

// Scoreboard Modal
async function showScoreboardModal() {
  const modal = document.querySelector("#scoreboard-modal");
  const body = document.querySelector("#scoreboard-body");
  modal?.classList.remove("hidden");
  body.innerHTML = `<div class="loading-spinner">Evaluating held-out SEVIR/proxy split...</div>`;

  try {
    const res = await fetch(`${API_BASE}/scoreboard`);
    const data = await res.json();

    const m = data.models || {};
    const vajra = m.VajraNow_ML || {};
    const lk = m.LucasKanade_OpticalFlow || {};
    const persist = m.Persistence || {};

    body.innerHTML = `
      <div style="font-size:11px;color:#94a3b8;margin-bottom:12px;">
        Dataset: <strong>${data.dataset}</strong> · Lead: <strong>T+${data.lead_time_minutes} min</strong> · Held-out Events: <strong>${data.holdout_events.join(", ")}</strong>
      </div>
      <table class="scoreboard-table">
        <thead>
          <tr>
            <th>Model Architecture</th>
            <th>CSI ↑</th>
            <th>POD ↑</th>
            <th>FAR ↓</th>
            <th>Bias ~1.0</th>
            <th>FSS (15km) ↑</th>
            <th>Brier ↓</th>
            <th>Brier Skill (BSS) ↑</th>
          </tr>
        </thead>
        <tbody>
          <tr class="highlight-row">
            <td>⚡ VajraNow AI/ML (U-Net + LGBM + Calib)</td>
            <td>${vajra.CSI}</td>
            <td>${vajra.POD}</td>
            <td>${vajra.FAR}</td>
            <td>${vajra.bias}</td>
            <td>${vajra.FSS_15km}</td>
            <td>${vajra.Brier}</td>
            <td>+${vajra.Brier_Skill_vs_Baseline}</td>
          </tr>
          <tr>
            <td>📐 Lucas-Kanade Semi-Lagrangian Flow</td>
            <td>${lk.CSI}</td>
            <td>${lk.POD}</td>
            <td>${lk.FAR}</td>
            <td>${lk.bias}</td>
            <td>${lk.FSS_15km}</td>
            <td>${lk.Brier}</td>
            <td>0.0000 (Ref)</td>
          </tr>
          <tr>
            <td>⏱️ Persistence Baseline</td>
            <td>${persist.CSI}</td>
            <td>${persist.POD}</td>
            <td>${persist.FAR}</td>
            <td>${persist.bias}</td>
            <td>${persist.FSS_15km}</td>
            <td>${persist.Brier}</td>
            <td>${persist.Brier_Skill_vs_Baseline}</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:16px;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);padding:10px 14px;border-radius:6px;">
        <strong style="color:#38bdf8;font-size:11px;">⚡ Advance Lead-Time to First Lightning Strike:</strong>
        <div style="font-size:11px;color:#cbd5e1;margin-top:4px;">
          Mean Advance Warning: <strong>${data.lead_time_to_first_strike?.mean_advance_warning_minutes} minutes</strong> before first strike detected (Median: ${data.lead_time_to_first_strike?.median_advance_warning_minutes} min across ${data.lead_time_to_first_strike?.sample_cells} convective cells).
        </div>
      </div>
    `;
  } catch (err) {
    body.innerHTML = `<div class="empty-state">Failed to load scoreboard data: ${err}</div>`;
  }
}

function hideScoreboardModal() {
  document.querySelector("#scoreboard-modal")?.classList.add("hidden");
}

window.addEventListener("DOMContentLoaded", initApp);
