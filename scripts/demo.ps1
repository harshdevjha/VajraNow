$ErrorActionPreference = 'Stop'
$base = if ($env:VAJRANOW_URL) { $env:VAJRANOW_URL.TrimEnd('/') } else { 'http://localhost:8000' }
Write-Host "========================================================="
Write-Host "  VAJRANOW: AI/ML Thunderstorm & Lightning Nowcaster"
Write-Host "  Problem Statement 26072 - End-to-End Demo"
Write-Host "  Target: $base"
Write-Host "========================================================="

Write-Host "`n[Step 1] Check System Health & Multimodal Engine Status"
Invoke-RestMethod "$base/health" | ConvertTo-Json -Depth 4

Write-Host "`n[Step 2] Query Active Convective Cells & Motion Vectors"
Invoke-RestMethod "$base/cells" | ConvertTo-Json -Depth 4

Write-Host "`n[Step 3] Query AI/ML Lightning Nowcast (Lead T+30 min)"
Invoke-RestMethod "$base/forecast?product=lightning&lead=30" | ConvertTo-Json -Depth 5

Write-Host "`n[Step 4] Test Radar Dropout Live Graceful Degradation (Disable Radar 1 Pune)"
Invoke-RestMethod "$base/radar/toggle?radar_id=radar_0&enabled=false" -Method Post | ConvertTo-Json -Depth 4
Invoke-RestMethod "$base/forecast?product=lightning&lead=30" | ConvertTo-Json -Depth 5
# Re-enable radar 1
Invoke-RestMethod "$base/radar/toggle?radar_id=radar_0&enabled=true" -Method Post | Out-Null

Write-Host "`n[Step 5] Inspect Explainability 'Why' Card (SHAP + Modality Ablation Attribution)"
Invoke-RestMethod "$base/alerts/SYN-001/why" | ConvertTo-Json -Depth 6

Write-Host "`n[Step 6] Query Administrative Block-Level Exposure Risk (GeoJSON)"
Invoke-RestMethod "$base/risk/blocks" | ConvertTo-Json -Depth 4

Write-Host "`n[Step 7] Inspect OASIS CAP 1.2 XML Alert Document"
(Invoke-WebRequest "$base/cap/SYN-001.xml").Content

Write-Host "`n[Step 8] Test Strict No-Leakage Hindcast Replay Session"
Invoke-RestMethod "$base/replay/start" -Method Post | ConvertTo-Json -Depth 4
Invoke-RestMethod "$base/replay/step" -Method Post | ConvertTo-Json -Depth 4

Write-Host "`n[Step 9] Inspect Honest Held-Out Evaluation Scoreboard"
Invoke-RestMethod "$base/scoreboard" | ConvertTo-Json -Depth 6

Write-Host "`n========================================================="
Write-Host "  Demo Completed Successfully! Open $base in browser."
Write-Host "========================================================="
