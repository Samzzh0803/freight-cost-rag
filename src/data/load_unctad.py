"""
UNCTAD US.TransportCosts — per-unit freight rates (USD/kg) by destination and mode.

Covers 2016-2021. USAID training data is 2006-2015, so there's no year overlap.
We use this purely as a structural benchmark: attach a lane-level $/kg signal to each
USAID row and let the model decide how much weight it deserves.

One known quirk: UNCTAD measures US *import* rates (goods shipped TO the US).
USAID ships the other direction. The rate still encodes lane difficulty,
just with inverted directionality — see notebook 02_eda_unctad for the full discussion.
"""

import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = _PROJECT_ROOT / "data" / "raw" / "unctad" / "unctad_transport_costs_bilateral.csv"

# UNCTAD uses different label conventions than USAID
MODE_MAP = {
    "Air": "Air",
    "Sea": "Ocean",
    "Road": "Truck",
    "Railway": "Rail",
}

VALID_MODES = set(MODE_MAP.keys())

# UNCTAD uses UN-style country names; USAID uses shorter versions
DEST_NORM = {
    "Congo, Dem. Rep. of the": "Congo, DRC",
    "Tanzania, United Republic of": "Tanzania",
    "Viet Nam": "Vietnam",
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(CSV_FILE)
    df = df.rename(columns={
        "TransportMode_Label": "mode_unctad",
        "Destination_Label": "destination",
        "Year": "year",
        "Perunit_freight_rate_USkg_Value": "rate_usd_per_kg",
    })
    df = df[df["mode_unctad"].isin(VALID_MODES)].copy()
    df["mode"] = df["mode_unctad"].map(MODE_MAP)
    df = df.dropna(subset=["rate_usd_per_kg"])
    df["destination"] = df["destination"].replace(DEST_NORM)
    return df[["destination", "mode", "year", "rate_usd_per_kg"]]


def build_lookup(df: pd.DataFrame) -> dict:
    """Build a fast lookup dict: (destination, mode) → median $/kg across all years.

    Fallback stored under __mode_fallback__: if a destination isn't in UNCTAD,
    we return the global mode median rather than NaN. LightGBM handles true NaN
    (mode = "Other") natively so no imputation needed there.
    """
    grouped = (
        df.groupby(["destination", "mode"])["rate_usd_per_kg"]
        .median()
    )
    lookup = grouped.to_dict()

    # Pre-compute mode-level fallbacks
    mode_fallback = df.groupby("mode")["rate_usd_per_kg"].median().to_dict()
    lookup["__mode_fallback__"] = mode_fallback

    return lookup


def get_rate(lookup: dict, destination: str, mode: str) -> float:
    """Look up $/kg rate with fallback to mode-level median."""
    rate = lookup.get((destination, mode))
    if rate is not None:
        return rate
    return lookup.get("__mode_fallback__", {}).get(mode, float("nan"))


def load_lookup() -> dict:
    return build_lookup(load_raw())
