"""
Load and clean the USAID SCMS Delivery History dataset.

Cleaning decisions worth knowing:
- ~4,126 rows have non-numeric Freight Cost — these are "From RDC" shipments where
  freight is bundled into the commodity price. Not separately priced, so not modellable.
  Dropped on numeric coercion.
- Air Charter collapsed into Air — same mechanism, different procurement arrangement.
- INCO terms bucketed into 4 groups (ExWorks, DDP, CIF, RDC) to reduce cardinality.
- Cost and weight capped at 99th percentile — a handful of extreme outliers swamp the
  log-scale target otherwise.
- Time split at 2013-12-01 (~80/20). Never random-split temporal data.
- log_cost = log1p(Freight_Cost_USD) is the model target.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = _PROJECT_ROOT / "data" / "raw" / "usaid" / "SCMS_Delivery_History_Dataset_20150929.csv"

MODE_MAP = {
    "Air": "Air",
    "Air Charter": "Air",
    "Truck": "Truck",
    "Ocean": "Ocean",
}

INCO_MAP = {
    "EXW": "ExWorks",
    "FCA": "ExWorks",
    "DDP": "DDP",
    "DDU": "DDP",
    "DAP": "DDP",
    "CIP": "CIF",
    "CIF": "CIF",
    "N/A - From RDC": "RDC",
}


def _extract_country_from_site(site: str) -> str:
    """Pull origin country from Manufacturing Site (e.g. 'Cipla Ltd, Mumbai, India' → 'India').
    Heuristic — last comma segment. Returns '' for unparseable values."""
    if not isinstance(site, str):
        return ""
    parts = [p.strip().rstrip(".") for p in site.split(",")]
    return parts[-1] if len(parts) > 1 else ""


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, encoding="latin-1")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numeric coercion — drops ~4,126 rows with text freight values
    df["Freight_Cost_USD"] = pd.to_numeric(df["Freight Cost (USD)"], errors="coerce")
    df["Weight_kg"] = pd.to_numeric(df["Weight (Kilograms)"], errors="coerce")
    df = df.dropna(subset=["Freight_Cost_USD", "Weight_kg"])

    # Drop zero or negative freight costs (not modellable)
    df = df[df["Freight_Cost_USD"] > 0]

    # Outlier capping at 99th percentile
    cap_cost = df["Freight_Cost_USD"].quantile(0.99)
    cap_weight = df["Weight_kg"].quantile(0.99)
    df["Freight_Cost_USD"] = df["Freight_Cost_USD"].clip(upper=cap_cost)
    df["Weight_kg"] = df["Weight_kg"].clip(upper=cap_weight)

    # Mode grouping
    df["mode"] = df["Shipment Mode"].map(MODE_MAP).fillna("Other")

    # INCO term grouping
    df["inco_group"] = df["Vendor INCO Term"].map(INCO_MAP).fillna("Other")

    # Parse date
    df["delivery_date"] = pd.to_datetime(df["Scheduled Delivery Date"], dayfirst=True, errors="coerce")
    df["year"] = df["delivery_date"].dt.year

    # Origin country from Manufacturing Site
    df["origin_country"] = df["Manufacturing Site"].apply(_extract_country_from_site)

    # Rename key columns
    df = df.rename(columns={
        "Country": "dest_country",
        "Product Group": "product_group",
        "Sub Classification": "sub_class",
    })

    # Log target
    df["log_cost"] = np.log1p(df["Freight_Cost_USD"])

    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    return clean(load_raw())


def time_split(df: pd.DataFrame, split_date: str = "2013-12-01"):
    """Time-based 80/20 split. NEVER use random split on temporal data."""
    mask = df["delivery_date"] < pd.Timestamp(split_date)
    return df[mask].copy(), df[~mask].copy()
