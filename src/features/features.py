"""
Feature engineering: USAID base features + UNCTAD enrichment join.

Feature set:
  log_weight          continuous  log1p(Weight_kg) — log-scale weight mirrors log-scale cost
  weight_bucket       int         quantile bucket (0-4) of log_weight — captures non-linearity
  mode_encoded        int         Air=0, Ocean=1, Truck=2, Other=3
  inco_encoded        int         ExWorks=0, DDP=1, CIF=2, RDC=3, Other=4
  product_group_enc   int         label-encoded product_group (no natural ordering — trees ignore
                                  integer magnitude; LinearRegression baseline treats it as
                                  continuous, which is wrong — interpret baseline coefficients
                                  for this feature with caution)
  year                int         delivery year, capped at MAX_TRAIN_YEAR for inference
  route_label         int         label-encoded (origin_country, dest_country) pair;
                                  unknown routes at inference → -1 (LightGBM routes to NaN leaf)
  unctad_rate_usd_kg  float       UNCTAD $/kg benchmark; NaN if no coverage — LightGBM handles

Inference notes:
  year is capped at MAX_TRAIN_YEAR (2015) — tree models assign any year ≥ 2015 to the
  same leaf, so values beyond the training range predict identically anyway. Cap is
  explicit rather than silent.

  route_label uses a plain dict (not LabelEncoder) so unknown routes return -1 safely.
  LightGBM will route -1 to the best available child, not crash.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

MODE_ORDER = ["Air", "Ocean", "Truck", "Other"]
INCO_ORDER = ["ExWorks", "DDP", "CIF", "RDC", "Other"]

N_WEIGHT_BUCKETS = 5
MAX_TRAIN_YEAR = 2015  # training data ceiling; cap year at inference to avoid silent extrapolation


def get_weight_bin_edges(df: pd.DataFrame) -> np.ndarray:
    """Compute weight_bucket bin edges from df (call on train split only)."""
    log_w = np.log1p(df["Weight_kg"]).clip(lower=0)
    edges = np.nanquantile(log_w, np.linspace(0, 1, N_WEIGHT_BUCKETS + 1))
    return np.unique(edges)


def build_route_map(df: pd.DataFrame) -> dict:
    """Build route → integer mapping from a dataframe (call on train split only)."""
    route_series = df["origin_country"].fillna("Unknown") + " → " + df["dest_country"].fillna("Unknown")
    unique_routes = sorted(route_series.unique())
    return {r: i for i, r in enumerate(unique_routes)}


def build_features(
    df: pd.DataFrame,
    unctad_lookup: dict | None = None,
    route_map: dict | None = None,
    weight_bin_edges: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Transform cleaned USAID df into model-ready feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_usaid.clean().
    unctad_lookup : dict | None
        Output of load_unctad.build_lookup(). Pass None to skip enrichment.
    route_map : dict | None
        Output of build_route_map(df_train). Pass None during training (map
        built from df). Pass the training map at inference so unknown routes
        return -1 instead of getting a new code.
    weight_bin_edges : np.ndarray | None
        Bin edges from get_weight_bin_edges(df_train). Pass None during training
        (edges computed from df). Pass the training edges for test/inference so
        both splits use identical bucket boundaries — prevents leakage.

    Returns
    -------
    pd.DataFrame with columns defined in feature_names().
    """
    out = pd.DataFrame(index=df.index)

    # Weight features — bin edges fit on train only to prevent leakage
    out["log_weight"] = np.log1p(df["Weight_kg"])
    log_w = out["log_weight"].clip(lower=0)
    if weight_bin_edges is None:
        weight_bin_edges = get_weight_bin_edges(df)
    out["weight_bucket"] = pd.cut(
        log_w,
        bins=weight_bin_edges,
        labels=False,
        include_lowest=True,
    ).astype("Int64")

    # Mode encoding (fixed mapping — stable across train/test, no fitting required)
    mode_map = {m: i for i, m in enumerate(MODE_ORDER)}
    out["mode_encoded"] = df["mode"].map(mode_map).fillna(len(MODE_ORDER) - 1).astype(int)

    # INCO encoding (fixed mapping)
    inco_map = {t: i for i, t in enumerate(INCO_ORDER)}
    out["inco_encoded"] = df["inco_group"].map(inco_map).fillna(len(INCO_ORDER) - 1).astype(int)

    # Product group encoding — label-encoded integer, no natural ordering.
    # Trees (LightGBM) treat it as categorical and ignore integer magnitude.
    # LinearRegression will treat it as continuous — baseline coefficients for this
    # feature are not interpretable as a direction.
    le_pg = LabelEncoder()
    out["product_group_enc"] = le_pg.fit_transform(df["product_group"].fillna("Unknown"))

    # Year — capped at MAX_TRAIN_YEAR so inference values beyond training range
    # don't extrapolate silently (tree assigns them to last leaf anyway, but
    # the cap makes the assumption explicit and keeps the feature in-distribution)
    out["year"] = df["year"].fillna(df["year"].median()).astype(int).clip(upper=MAX_TRAIN_YEAR)

    # Route label: (origin_country, dest_country) as integer.
    # Dict-based (not LabelEncoder) so unknown routes at inference safely return -1.
    # Pass a pre-built route_map (from build_route_map(df_train)) at inference time;
    # if None, the map is built from the current df (training mode).
    route_series = df["origin_country"].fillna("Unknown") + " → " + df["dest_country"].fillna("Unknown")
    if route_map is None:
        route_map = build_route_map(df)
    out["route_label"] = route_series.map(route_map).fillna(-1).astype(int)

    # UNCTAD enrichment
    if unctad_lookup is not None:
        from src.data.load_unctad import get_rate
        out["unctad_rate_usd_kg"] = [
            get_rate(unctad_lookup, row["dest_country"], row["mode"])
            for _, row in df[["dest_country", "mode"]].iterrows()
        ]
    else:
        out["unctad_rate_usd_kg"] = float("nan")

    return out


def get_target(df: pd.DataFrame) -> pd.Series:
    """Log-transformed freight cost. This is y for all models."""
    return np.log1p(df["Freight_Cost_USD"])


def feature_names() -> list[str]:
    return [
        "log_weight",
        "weight_bucket",
        "mode_encoded",
        "inco_encoded",
        "product_group_enc",
        "year",
        "route_label",
        "unctad_rate_usd_kg",
    ]


# LightGBM categorical feature names — pass to categorical_feature param in training
LGBM_CAT_FEATURES = ["mode_encoded", "inco_encoded", "product_group_enc", "route_label"]
