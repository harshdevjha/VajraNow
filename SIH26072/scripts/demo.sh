#!/usr/bin/env bash
set -e

BASE_URL="${VAJRANOW_URL:-http://localhost:8000}"
echo "========================================================="
echo "  VAJRANOW: AI/ML Thunderstorm & Lightning Nowcaster"
echo "  Problem Statement 26072 - End-to-End Demo"
echo "  Target: $BASE_URL"
echo "========================================================="

echo -e "\n[Step 1] System Health"
curl -s "$BASE_URL/health"

echo -e "\n\n[Step 2] Active Cells"
curl -s "$BASE_URL/cells"

echo -e "\n\n[Step 3] AI/ML Lightning Forecast T+30m"
curl -s "$BASE_URL/forecast?product=lightning&lead=30"

echo -e "\n\n[Step 4] Radar Dropout Test (Disable Radar 0)"
curl -s -X POST "$BASE_URL/radar/toggle?radar_id=radar_0&enabled=false"
curl -s "$BASE_URL/forecast?product=lightning&lead=30"
curl -s -X POST "$BASE_URL/radar/toggle?radar_id=radar_0&enabled=true" > /dev/null

echo -e "\n\n[Step 5] Explainability 'Why' Card (SHAP & Modality Ablation)"
curl -s "$BASE_URL/alerts/SYN-001/why"

echo -e "\n\n[Step 6] Block Exposure Risk"
curl -s "$BASE_URL/risk/blocks"

echo -e "\n\n[Step 7] CAP 1.2 XML Alert"
curl -s "$BASE_URL/cap/SYN-001.xml"

echo -e "\n\n[Step 8] Hindcast Replay Session"
curl -s -X POST "$BASE_URL/replay/start"
curl -s -X POST "$BASE_URL/replay/step"

echo -e "\n\n[Step 9] Scoreboard"
curl -s "$BASE_URL/scoreboard"

echo -e "\n\n========================================================="
echo "  Demo Completed! Open $BASE_URL in your browser."
echo "========================================================="
