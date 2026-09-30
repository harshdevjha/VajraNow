from typing import Any

import numpy as np

from vajranow.config import load_config

# Representative administrative blocks across the domain bbox [73.0, 18.0, 79.0, 23.0]
DOMAIN_BLOCKS = [
    {
        "block_id": "MH-PUN-01",
        "name": "Pune Metro / Haveli",
        "district": "Pune",
        "population_density": 9500,  # persons / km2
        "land_cover": "urban",
        "bounds": [73.7, 18.4, 74.0, 18.7],
    },
    {
        "block_id": "MH-PUN-02",
        "name": "Shirur Agricultural Belt",
        "district": "Pune",
        "population_density": 380,
        "land_cover": "agricultural",
        "bounds": [74.1, 18.6, 74.5, 18.9],
    },
    {
        "block_id": "MH-NGP-01",
        "name": "Nagpur Urban Center",
        "district": "Nagpur",
        "population_density": 5800,
        "land_cover": "urban",
        "bounds": [78.9, 21.0, 79.2, 21.3],
    },
    {
        "block_id": "MH-NGP-02",
        "name": "Hingna Forest Corridor",
        "district": "Nagpur",
        "population_density": 210,
        "land_cover": "forest",
        "bounds": [78.6, 20.8, 79.0, 21.1],
    },
    {
        "block_id": "MH-SOL-01",
        "name": "Solapur North",
        "district": "Solapur",
        "population_density": 720,
        "land_cover": "agricultural",
        "bounds": [75.7, 17.6, 76.1, 18.0],
    },
    {
        "block_id": "MH-AUR-01",
        "name": "Chhatrapati Sambhajinagar (Aurangabad)",
        "district": "Aurangabad",
        "population_density": 3400,
        "land_cover": "urban",
        "bounds": [75.2, 19.7, 75.5, 20.0],
    },
    {
        "block_id": "TS-HYD-01",
        "name": "Hyderabad Outer Conurbation",
        "district": "Hyderabad",
        "population_density": 8200,
        "land_cover": "urban",
        "bounds": [78.3, 17.3, 78.7, 17.6],
    },
]


class RiskCalculator:
    """Calculates exposure-weighted thunderstorm risk aggregated to administrative blocks.

    Formula:
    Exposure_Weight = (w_pop * norm(population_density) + w_land * land_multiplier)
    Risk = P(hazard) * Exposure_Weight
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

    def calculate_block_risks(self, cells_data: list[dict]) -> dict[str, Any]:
        """Compute block-level exposure and risk scores from current storm cell locations."""
        cfg = self.config.get("exposure", {})
        pop_w = cfg.get("population_weight", 1.0)
        land_w = cfg.get("land_cover_weight", 0.5)

        multipliers = {
            "urban": cfg.get("urban_multiplier", 1.6),
            "agricultural": cfg.get("rural_multiplier", 1.0),
            "forest": cfg.get("forest_multiplier", 0.6),
            "water": cfg.get("water_multiplier", 0.2),
        }

        features = []
        for block in DOMAIN_BLOCKS:
            min_lon, min_lat, max_lon, max_lat = block["bounds"]
            center_lon = (min_lon + max_lon) / 2.0
            center_lat = (min_lat + max_lat) / 2.0

            # Calculate proximity to active convective cells
            cell_threats = []
            for cell in cells_data:
                coords = cell.get("coordinates")
                if not coords:
                    continue
                c_lon, c_lat = coords
                # Approximate distance in km
                d_km = np.hypot((c_lon - center_lon) * 105.0, (c_lat - center_lat) * 111.0)
                p_strike = cell.get("p_lightning", cell.get("probability", 0.3))

                # Distance decay kernel (15 km danger radius)
                threat = p_strike * np.exp(-max(0.0, d_km - 5.0) / 18.0)
                cell_threats.append(threat)

            max_p = max(cell_threats) if cell_threats else 0.05
            mean_p = float(np.mean(cell_threats)) if cell_threats else 0.02

            # Compute exposure factor
            pop_norm = min(1.0, block["population_density"] / 10000.0)
            land_mult = multipliers.get(block["land_cover"], 1.0)
            exposure_score = float(pop_w * pop_norm + land_w * (land_mult / 2.0))

            # Exposure-weighted risk
            max_risk = round(float(max_p * exposure_score), 4)
            mean_risk = round(float(mean_p * exposure_score), 4)

            # Categorize alert tier
            if max_risk >= 0.50:
                risk_tier = "CRITICAL"
                color = "#dc2626"
            elif max_risk >= 0.25:
                risk_tier = "HIGH"
                color = "#ea580c"
            elif max_risk >= 0.12:
                risk_tier = "MODERATE"
                color = "#eab308"
            else:
                risk_tier = "LOW"
                color = "#22c55e"

            polygon_coords = [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ]

            features.append(
                {
                    "type": "Feature",
                    "id": block["block_id"],
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": polygon_coords,
                    },
                    "properties": {
                        "block_id": block["block_id"],
                        "name": block["name"],
                        "district": block["district"],
                        "population_density": block["population_density"],
                        "land_cover": block["land_cover"],
                        "max_hazard_prob": round(float(max_p), 3),
                        "exposure_weight": round(exposure_score, 3),
                        "max_risk_score": max_risk,
                        "mean_risk_score": mean_risk,
                        "risk_tier": risk_tier,
                        "risk_color": color,
                    },
                }
            )

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "population_weight": pop_w,
                "land_cover_weight": land_w,
                "domain_blocks_evaluated": len(DOMAIN_BLOCKS),
                "status": "prototype, not operational",
            },
        }
