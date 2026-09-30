from datetime import UTC, datetime
from xml.etree import ElementTree as ET

from vajranow.cap import CAP_NS, create_cap


def test_cap_output_is_well_formed_and_has_required_core_fields():
    document = create_cap("demo-1", "Prototype alert", "Not operational", datetime(2026, 1, 1, tzinfo=UTC))
    root = ET.fromstring(document)
    assert root.tag == f"{{{CAP_NS}}}alert"
    assert root.findtext(f"{{{CAP_NS}}}identifier") == "demo-1"
    assert root.findtext(f"{{{CAP_NS}}}status") == "Test"
    assert root.find(f"{{{CAP_NS}}}info/{{{CAP_NS}}}event").text == "Thunderstorm"
