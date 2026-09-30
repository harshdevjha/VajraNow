from datetime import UTC, datetime, timedelta
from xml.etree import ElementTree as ET

import numpy as np

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
ET.register_namespace("", CAP_NS)


def create_cap(
    alert_id: str,
    headline: str,
    description: str,
    sent: datetime | None = None,
    *,
    expires_minutes: int = 45,
    severity: str = "Severe",
    urgency: str = "Expected",
    certainty: str = "Likely",
    coordinates: list[float] | None = None,
    radius_km: float = 15.0,
) -> str:
    """Generate a fully OASIS-compliant Common Alerting Protocol (CAP 1.2) XML document."""
    sent_dt = sent or datetime.now(UTC)
    expires_dt = sent_dt + timedelta(minutes=expires_minutes)

    sent_str = sent_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
    expires_str = expires_dt.isoformat(timespec="seconds").replace("+00:00", "Z")

    root = ET.Element(f"{{{CAP_NS}}}alert")

    ET.SubElement(root, f"{{{CAP_NS}}}identifier").text = alert_id
    ET.SubElement(root, f"{{{CAP_NS}}}sender").text = "vajranow-forecast-engine@imd.gov.in"
    ET.SubElement(root, f"{{{CAP_NS}}}sent").text = sent_str
    ET.SubElement(root, f"{{{CAP_NS}}}status").text = "Test"  # Prototype / Test
    ET.SubElement(root, f"{{{CAP_NS}}}msgType").text = "Alert"
    ET.SubElement(root, f"{{{CAP_NS}}}scope").text = "Public"

    info = ET.SubElement(root, f"{{{CAP_NS}}}info")
    ET.SubElement(info, f"{{{CAP_NS}}}category").text = "Met"
    ET.SubElement(info, f"{{{CAP_NS}}}event").text = "Thunderstorm"
    ET.SubElement(info, f"{{{CAP_NS}}}urgency").text = urgency
    ET.SubElement(info, f"{{{CAP_NS}}}severity").text = severity
    ET.SubElement(info, f"{{{CAP_NS}}}certainty").text = certainty
    ET.SubElement(info, f"{{{CAP_NS}}}expires").text = expires_str
    ET.SubElement(info, f"{{{CAP_NS}}}headline").text = headline
    ET.SubElement(info, f"{{{CAP_NS}}}description").text = description
    ET.SubElement(
        info, f"{{{CAP_NS}}}instruction"
    ).text = "Take immediate shelter indoors. Avoid open fields, metal structures, and tall isolated trees. Prototype output."

    # Parameter for engine tag
    param = ET.SubElement(info, f"{{{CAP_NS}}}parameter")
    ET.SubElement(param, f"{{{CAP_NS}}}valueName").text = "ForecastEngine"
    ET.SubElement(param, f"{{{CAP_NS}}}value").text = "VajraNow AI/ML Multi-Radar Nowcaster"

    # Area tag
    area = ET.SubElement(info, f"{{{CAP_NS}}}area")
    ET.SubElement(area, f"{{{CAP_NS}}}areaDesc").text = f"Cell {alert_id} Convective Impact Zone"

    if coordinates and len(coordinates) == 2:
        lon, lat = coordinates
        angles = np.linspace(0, 2 * np.pi, 9)
        d_deg_lat = radius_km / 111.0
        d_deg_lon = radius_km / (111.0 * np.cos(np.radians(lat)) + 1e-5)
        poly_pts = [f"{lat + d_deg_lat * np.sin(a):.4f},{lon + d_deg_lon * np.cos(a):.4f}" for a in angles]
        ET.SubElement(area, f"{{{CAP_NS}}}polygon").text = " ".join(poly_pts)
        ET.SubElement(area, f"{{{CAP_NS}}}circle").text = f"{lat:.4f},{lon:.4f} {radius_km:.1f}"

    return ET.tostring(root, encoding="unicode", xml_declaration=True)
