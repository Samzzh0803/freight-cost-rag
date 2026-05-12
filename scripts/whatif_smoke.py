"""Smoke script for the Week 9 what-if chat path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.whatif import whatif_chat


def main() -> None:
    shipment = {
        "Weight_kg": 850.0,
        "mode": "Air",
        "inco_group": "ExWorks",
        "product_group": "ARV",
        "year": 2014,
        "origin_country": "India",
        "dest_country": "Kenya",
    }

    messages = [
        "What if we switch this shipment to ocean?",
        "Ship it to Nigeria instead",
        "Increase the weight to 1250 kg",
    ]

    for message in messages:
        print(f"\n=== {message} ===")
        try:
            result = whatif_chat(message, shipment)
        except Exception as exc:
            result = {"message": message, "error": str(exc)}
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
