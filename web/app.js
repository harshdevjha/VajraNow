// VajraNow Front-End Application Controller - Executive Workbench Edition
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
let currentBasemap = "terrain"; // "terrain", "satellite", "roadmap"

// Google Maps Tile Sources
const GOOGLE_TILE_SOURCES = {
  terrain: "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
  satellite: "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
  roadmap: "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
};

// Radar Sites (Pune, Nagpur, Hyderabad)
const RADAR_SITES = [
  { id: "radar_0", name: "Pune DWR", center: [73.8567, 18.5204], radiusKm: 120 },
  { id: "radar_1", name: "Nagpur DWR", center: [79.0882, 21.1458], radiusKm: 130 },
  { id: "radar_2", name: "Hyderabad DWR", center: [78.4867, 17.385], radiusKm: 120 },
];

// Helper to create circle polygon in GeoJSON
function createGeoJsonCircle(center, radiusInKm, points = 64) {
  const coords = { latitude: center[1], longitude: center[0] };
  const km = radiusInKm;
  const ret = [];
  const distanceX = km / (111.32 * Math.cos((coords.latitude * Math.PI) / 180));
  const distanceY = km / 110.574;

  for (let i = 0; i < points; i++) {
    const theta = (i / points) * (2 * Math.PI);
    const x = distanceX * Math.cos(theta);
    const y = distanceY * Math.sin(theta);
    ret.push([coords.longitude + x, coords.latitude + y]);
  }
  ret.push(ret[0]);
  return {
    type: "Feature",
    properties: { radius: radiusInKm },
    geometry: {
      type: "Polygon",
      coordinates: [ret],
    },
  };
}

// Generate Radar Range Rings GeoJSON
function getRadarRingsGeoJson() {
  const features = [];
  RADAR_SITES.forEach((site) => {
    const isOnline = radarStates[site.id] !== false;
    const ring = createGeoJsonCircle(site.center, site.radiusKm);
    ring.properties = {
      radar_id: site.id,
      name: site.name,
      online: isOnline,
      color: isOnline ? "#0284c7" : "#94a3b8",
    };
    features.push(ring);
  });
  return { type: "FeatureCollection", features };
}

// Generate Radar Station Markers GeoJSON
function getRadarStationsGeoJson() {
  const features = RADAR_SITES.map((site) => {
    const isOnline = radarStates[site.id] !== false;
    return {
      type: "Feature",
      properties: {
        radar_id: site.id,
        name: site.name,
        online: isOnline,
      },
      geometry: {
        type: "Point",
        coordinates: site.center,
      },
    };
  });
  return { type: "FeatureCollection", features };
}

// Generate Domain Bounding Box Polygon GeoJSON
function getDomainBboxGeoJson(bbox) {
  if (!bbox) bbox = [73.0, 18.0, 79.0, 23.0];
  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: { name: "Central India Nowcast Grid (~2 km)" },
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [bbox[0], bbox[1]],
              [bbox[2], bbox[1]],
              [bbox[2], bbox[3]],
              [bbox[0], bbox[3]],
              [bbox[0], bbox[1]],
            ],
          ],
        },
      },
    ],
  };
}

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

    // Auto resize map on window resize
    window.addEventListener("resize", () => {
      if (map) map.resize();
    });
  } catch (err) {
    console.error("Failed to initialize VajraNow console:", err);
  }
}

function setupUIEvents() {
  // Basemap Layer Switcher (Terrain / Satellite / Roadmap)
  document.querySelectorAll(".map-layer-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".map-layer-btn").forEach((b) => b.classList.remove("active"));
      e.target.classList.add("active");
      const layerKey = e.target.dataset.layer;
      switchBasemap(layerKey);
    });
  });

  // Map view preset buttons
  document.querySelector("#btn-view-india")?.addEventListener("click", (e) => {
    document.querySelectorAll(".map-preset-btn").forEach((p) => p.classList.remove("active"));
    e.currentTarget.classList.add("active");
    if (map) {
      map.flyTo({ center: [78.9629, 22.5937], zoom: 4.8, essential: true });
    }
  });

  document.querySelector("#btn-view-domain")?.addEventListener("click", (e) => {
    document.querySelectorAll(".map-preset-btn").forEach((p) => p.classList.remove("active"));
    e.currentTarget.classList.add("active");
    const bbox = config?.region?.bbox || [73.0, 18.0, 79.0, 23.0];
    if (map) {
      map.flyTo({ center: [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], zoom: 6.6, essential: true });
    }
  });

  document.querySelectorAll(".map-preset-btn.mini").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const radarKey = e.target.dataset.radar;
      const site = RADAR_SITES.find((s) => s.id.includes(radarKey) || s.name.toLowerCase().includes(radarKey));
      if (site && map) {
        map.flyTo({ center: site.center, zoom: 8.5, essential: true });
      }
    });
  });

  // Engine toggle
  document.querySelector("#mode-ml")?.addEventListener("click", () => setEngineMode("ML"));
  document.querySelector("#mode-baseline")?.addEventListener("click", () => setEngineMode("BASELINE"));

  // Product toggle
  document.querySelector("#prod-lightning")?.addEventListener("click", () => setProduct("lightning"));
  document.querySelector("#prod-severe")?.addEventListener("click", () => setProduct("severe"));

  // Refresh Button
  document.querySelector("#btn-refresh")?.addEventListener("click", () => {
    refreshData();
  });

  // Lead slider
  const ALLOWED_LEADS = [10, 20, 30, 60, 90, 120, 180];
  const slider = document.querySelector("#lead-slider");
  slider?.addEventListener("input", (e) => {
    const idx = parseInt(e.target.value, 10);
    currentLead = ALLOWED_LEADS[idx] !== undefined ? ALLOWED_LEADS[idx] : 30;
    document.querySelector("#lead-value").textContent = `T+${currentLead} min`;
    const summaryLeadEl = document.querySelector("#summary-lead-val");
    if (summaryLeadEl) summaryLeadEl.textContent = `T+${currentLead} min`;
    updateConfidenceBadge(currentLead);
    refreshData();
  });

  // Radar dropout toggles
  ["radar_0", "radar_1", "radar_2"].forEach((rId) => {
    const el = document.querySelector(`#toggle-${rId.replace("_", "-")}`);
    el?.addEventListener("change", async (e) => {
      radarStates[rId] = e.target.checked;
      await toggleRadar(rId, e.target.checked);
      updateRadarRings();
    });
  });

  // Scoreboard modal
  document.querySelector("#btn-open-scoreboard")?.addEventListener("click", showScoreboardModal);
  document.querySelector("#btn-close-scoreboard")?.addEventListener("click", hideScoreboardModal);
  document.querySelector("#scoreboard-modal")?.addEventListener("click", (e) => {
    if (e.target.id === "scoreboard-modal") hideScoreboardModal();
  });

  // Replay controls
  document.querySelector("#btn-open-replay")?.addEventListener("click", toggleReplayMode);
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
  if (!badge) return;

  badge.className = "conf-badge";
  if (lead <= 60) {
    badge.classList.add("conf-high");
    badge.textContent = "✓ High Confidence (Physical Scaling Valid)";
  } else if (lead <= 90) {
    badge.classList.add("conf-moderate");
    badge.textContent = "⚠️ Moderate Confidence (Convective Dispersion)";
  } else {
    badge.classList.add("conf-low");
    badge.textContent = "⏱️ Low Confidence (Skill Horizon Decaying)";
  }
}

function setEngineMode(mode) {
  currentEngineMode = mode;
  document.querySelector("#mode-ml")?.classList.toggle("active", mode === "ML");
  document.querySelector("#mode-baseline")?.classList.toggle("active", mode === "BASELINE");
  
  const text = document.querySelector("#engine-badge-text");
  if (text) {
    text.textContent = mode === "ML" ? "ML (Res U-Net + LGBM)" : "BASELINE (LK Optical Flow)";
  }
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
  const text = document.querySelector("#engine-badge-text");
  if (!text) return;

  if (engine === "ML") {
    text.textContent = "ML (Res U-Net + LGBM)";
    text.className = "t-val highlight";
  } else if (engine === "ML_DEGRADED") {
    text.textContent = "ML DEGRADED (Radar Missing)";
    text.className = "t-val";
    text.style.color = "#d97706";
  } else {
    text.textContent = "BASELINE (LK Optical Flow)";
    text.className = "t-val";
    text.style.color = "#0284c7";
  }
}

// Switch Google Basemap layer
function switchBasemap(layerKey) {
  currentBasemap = layerKey;
  if (!map) return;

  const tileUrl = GOOGLE_TILE_SOURCES[layerKey] || GOOGLE_TILE_SOURCES.terrain;
  if (map.getSource("google-base")) {
    // MapLibre allows updating raster tiles dynamically
    map.removeLayer("google-base-layer");
    map.removeSource("google-base");

    map.addSource("google-base", {
      type: "raster",
      tiles: [tileUrl],
      tileSize: 256,
      attribution: "© Google Maps · VajraNow ⚡",
    });

    // Add back as bottom layer
    map.addLayer(
      {
        id: "google-base-layer",
        type: "raster",
        source: "google-base",
      },
      "domain-bbox-fill"
    );
  }
}

// Map Initialization
function initMap() {
  const bbox = config?.region?.bbox || [73.0, 18.0, 79.0, 23.0];
  const indiaCenter = [78.9629, 22.5937];

  try {
    map = new maplibregl.Map({
      container: "map",
      center: indiaCenter,
      zoom: 4.8,
      minZoom: 3.5,
      maxZoom: 16,
      style: {
        version: 8,
        sources: {
          "google-base": {
            type: "raster",
            tiles: [GOOGLE_TILE_SOURCES.terrain],
            tileSize: 256,
            attribution: "© Google Maps · VajraNow ⚡",
          },
        },
        layers: [
          {
            id: "google-base-layer",
            type: "raster",
            source: "google-base",
          },
        ],
      },
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

    map.on("load", () => {
      // 1. Domain Bounding Box Layer
      map.addSource("domain-bbox", {
        type: "geojson",
        data: getDomainBboxGeoJson(bbox),
      });
      map.addLayer({
        id: "domain-bbox-fill",
        type: "fill",
        source: "domain-bbox",
        paint: {
          "fill-color": "#0284c7",
          "fill-opacity": 0.05,
        },
      });
      map.addLayer({
        id: "domain-bbox-line",
        type: "line",
        source: "domain-bbox",
        paint: {
          "line-color": "#0284c7",
          "line-width": 2,
          "line-dasharray": [4, 3],
          "line-opacity": 0.85,
        },
      });

      // 2. Radar Range Rings Layer
      map.addSource("radar-rings", {
        type: "geojson",
        data: getRadarRingsGeoJson(),
      });
      map.addLayer({
        id: "radar-rings-fill",
        type: "fill",
        source: "radar-rings",
        paint: {
          "fill-color": ["get", "color"],
          "fill-opacity": 0.08,
        },
      });
      map.addLayer({
        id: "radar-rings-line",
        type: "line",
        source: "radar-rings",
        paint: {
          "line-color": ["get", "color"],
          "line-width": 1.8,
          "line-opacity": 0.75,
          "line-dasharray": [3, 2],
        },
      });

      // 3. Radar Station Towers Layer
      map.addSource("radar-stations", {
        type: "geojson",
        data: getRadarStationsGeoJson(),
      });
      map.addLayer({
        id: "radar-stations-halo",
        type: "circle",
        source: "radar-stations",
        paint: {
          "circle-radius": 9,
          "circle-color": "#0284c7",
          "circle-opacity": 0.35,
        },
      });
      map.addLayer({
        id: "radar-stations-dot",
        type: "circle",
        source: "radar-stations",
        paint: {
          "circle-radius": 4.5,
          "circle-color": "#ffffff",
          "circle-stroke-width": 2,
          "circle-stroke-color": "#0284c7",
        },
      });

      // 4. Risk Blocks Layer
      map.addSource("risk-blocks", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "risk-blocks-fill",
        type: "fill",
        source: "risk-blocks",
        paint: {
          "fill-color": ["get", "risk_color"],
          "fill-opacity": 0.25,
        },
      });
      map.addLayer({
        id: "risk-blocks-border",
        type: "line",
        source: "risk-blocks",
        paint: {
          "line-color": ["get", "risk_color"],
          "line-width": 1.8,
          "line-opacity": 0.8,
        },
      });

      // 5. Motion Tracks Layer
      map.addSource("tracks", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "tracks-line",
        type: "line",
        source: "tracks",
        paint: {
          "line-color": "#0284c7",
          "line-width": 2.5,
          "line-dasharray": [3, 2],
          "line-opacity": 0.9,
        },
      });

      // 6. Cell Halos (Reflectivity / Risk glow)
      map.addSource("cells", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: "cell-halo",
        type: "circle",
        source: "cells",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "dbz"], 30, 18, 50, 44],
          "circle-color": [
            "interpolate",
            ["linear"],
            ["get", "dbz"],
            30, "#22c55e",
            38, "#eab308",
            45, "#f97316",
            55, "#dc2626",
          ],
          "circle-opacity": 0.45,
          "circle-blur": 0.5,
        },
      });

      // 7. Cell Core Point
      map.addLayer({
        id: "cell-core",
        type: "circle",
        source: "cells",
        paint: {
          "circle-radius": 7,
          "circle-color": [
            "case",
            [">=", ["get", "dbz"], 40], "#dc2626",
            [">=", ["get", "dbz"], 35], "#d97706",
            "#059669",
          ],
          "circle-stroke-width": 2,
          "circle-stroke-color": "#ffffff",
        },
      });

      // Cell Click for SHAP
      map.on("click", "cell-core", (e) => {
        if (e.features && e.features[0]) {
          const props = e.features[0].properties;
          loadWhyCard(props.cell_id || props.id);
        }
      });

      map.on("mouseenter", "cell-core", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "cell-core", () => {
        map.getCanvas().style.cursor = "";
      });
    });
  } catch (err) {
    console.warn("MapLibre initialization:", err);
  }
}

function updateRadarRings() {
  if (!map || !map.getSource) return;
  if (map.getSource("radar-rings")) {
    map.getSource("radar-rings").setData(getRadarRingsGeoJson());
  }
  if (map.getSource("radar-stations")) {
    map.getSource("radar-stations").setData(getRadarStationsGeoJson());
  }
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
    renderFleetTable(cellsData, fcData, alertsData);
    renderTriageQueue(alertsData, cellsData);
    renderRiskSummary(riskData);

    // Update Top KPI Metrics
    const cellCount = cellsData.cells ? cellsData.cells.length : 0;
    const kpiActiveCells = document.querySelector("#kpi-active-cells");
    if (kpiActiveCells) kpiActiveCells.textContent = cellCount;
    const fleetBadge = document.querySelector("#fleet-count-badge");
    if (fleetBadge) fleetBadge.textContent = cellCount;

    // High risk blocks
    const highRiskCount = (riskData.features || []).filter(
      (f) => f.properties.risk_tier === "CRITICAL" || f.properties.risk_tier === "HIGH"
    ).length;
    const kpiHighRisk = document.querySelector("#kpi-high-risk-blocks");
    if (kpiHighRisk) kpiHighRisk.textContent = highRiskCount;

    // Radar availability KPI
    const activeRadars = (radarStates.radar_0 ? 1 : 0) + (radarStates.radar_1 ? 1 : 0) + (radarStates.radar_2 ? 1 : 0);
    const kpiRatio = document.querySelector("#kpi-radar-ratio");
    if (kpiRatio) kpiRatio.innerHTML = `${Math.round((activeRadars / 3) * 100)}<small class="kpi-unit">%</small>`;
    
    const summaryRadar = document.querySelector("#summary-radar-val");
    if (summaryRadar) summaryRadar.textContent = `${activeRadars}/3 Radars Online`;

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

// Render Convective Cell Fleet Table
function renderFleetTable(cellsData, fcData, alertsData) {
  const tbody = document.querySelector("#cells-table-body");
  if (!tbody) return;

  const cells = cellsData.cells || [];
  if (cells.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state-box">No convective cells currently detected.</td></tr>`;
    return;
  }

  const alertsMap = {};
  (alertsData.alerts || []).forEach((a) => {
    alertsMap[a.id] = a;
  });

  tbody.innerHTML = cells
    .map((c) => {
      const alert = alertsMap[c.id];
      const pLightning = alert ? alert.probability : (c.p_lightning || 0.45);
      const pPercent = Math.round(pLightning * 100);
      
      let probClass = "prob-low";
      if (pPercent >= 60) probClass = "prob-high";
      else if (pPercent >= 35) probClass = "prob-med";

      const velStr = c.velocity_kmh ? `${c.velocity_kmh} km/h` : "28.5 km/h";

      return `
      <tr onclick="loadWhyCard('${c.id}')">
        <td><strong>⚡ ${c.id}</strong></td>
        <td><span class="prob-pill ${c.max_dbz >= 40 ? 'prob-high' : 'prob-med'}">${c.max_dbz || 42} dBZ</span></td>
        <td class="mono-text">${c.ir_cooling_rate || 4.2} K/10m</td>
        <td><strong>${c.strike_count || 0}</strong></td>
        <td class="mono-text">${velStr}</td>
        <td><span class="prob-pill ${probClass}">${pPercent}% Risk</span></td>
        <td>
          <a class="cap-btn" href="${alert ? alert.cap_url : `/cap/${c.id}.xml`}" target="_blank" onclick="event.stopPropagation()">📥 CAP 1.2</a>
        </td>
      </tr>
    `;
    })
    .join("");
}

// Render Triage Queue (Right Column)
function renderTriageQueue(alertsData, cellsData) {
  const triageListEl = document.querySelector("#triage-list");
  if (!triageListEl) return;

  const alerts = alertsData.alerts || [];
  if (alerts.length === 0) {
    triageListEl.innerHTML = `<div class="empty-state-box">No critical alerts currently breaching threshold (${alertsData.threshold_on || 0.55}).</div>`;
    return;
  }

  triageListEl.innerHTML = alerts
    .map((a) => {
      const pPercent = Math.round(a.probability * 100);
      return `
      <div class="triage-item ${selectedCellId === a.id ? 'active' : ''}" onclick="loadWhyCard('${a.id}')">
        <div class="triage-cell-meta">
          <strong>⚡ Cell ${a.id}</strong>
          <small>${a.max_dbz} dBZ · ${a.strike_count} strikes · ${a.ir_cooling_rate} K cooling</small>
        </div>
        <div class="triage-score">${pPercent}%</div>
      </div>
    `;
    })
    .join("");
}

// SHAP TreeExplainer & Modality Ablation "Why" Card
async function loadWhyCard(cellId) {
  selectedCellId = cellId;
  const contentEl = document.querySelector("#why-card-content");
  if (!contentEl) return;

  contentEl.innerHTML = `<div class="empty-state-box">Evaluating SHAP feature attributions for ${cellId}...</div>`;

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
        
        <div class="why-header" style="margin-top:8px;">
          <span class="why-title" style="color:#d97706;">Modality Ablation Sensitivity</span>
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
      </div>
    `;

    // Highlight row in triage queue if present
    document.querySelectorAll(".triage-item").forEach((el) => {
      el.classList.toggle("active", el.textContent.includes(cellId));
    });

  } catch (err) {
    contentEl.innerHTML = `<div class="empty-state-box">No explanation data for ${cellId}.</div>`;
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
          <div style="font-size:10px;color:#5e776e;">Pop: ${p.population_density}/km² · ${p.land_cover}</div>
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
    document.querySelector("#replay-time-label").textContent = `T+${currentMinute} min (Replay)`;
    refreshData();

    if (data.is_finished) {
      clearInterval(replayTimer);
      isReplayRunning = false;
      document.querySelector("#btn-replay-play").textContent = "▶ Done";
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
  body.innerHTML = `<div class="loading-spinner" style="text-align:center;padding:16px;">Evaluating held-out SEVIR/proxy split...</div>`;

  try {
    const res = await fetch(`${API_BASE}/scoreboard`);
    const data = await res.json();

    const m = data.models || {};
    const vajra = m.VajraNow_ML || {};
    const lk = m.LucasKanade_OpticalFlow || {};
    const persist = m.Persistence || {};

    body.innerHTML = `
      <div style="font-size:11.5px;color:#5e776e;margin-bottom:12px;">
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
      <div style="margin-top:16px;background:#ecfdf5;border:1px solid #d1fae5;padding:12px 16px;border-radius:6px;">
        <strong style="color:#059669;font-size:12px;">⚡ Advance Warning Lead-Time to First Strike:</strong>
        <div style="font-size:11.5px;color:#0f1c18;margin-top:4px;">
          Mean Advance Warning: <strong>${data.lead_time_to_first_strike?.mean_advance_warning_minutes} minutes</strong> before first strike detection (Median: ${data.lead_time_to_first_strike?.median_advance_warning_minutes} min across ${data.lead_time_to_first_strike?.sample_cells} convective cells).
        </div>
      </div>
    `;
  } catch (err) {
    body.innerHTML = `<div class="empty-state-box">Failed to load scoreboard data: ${err}</div>`;
  }
}

function hideScoreboardModal() {
  document.querySelector("#scoreboard-modal")?.classList.add("hidden");
}

window.addEventListener("DOMContentLoaded", initApp);
