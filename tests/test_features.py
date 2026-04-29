"""
Tests for src/features/features.py.

Checks: correct shape, expected columns, no unexpected NaN in non-UNCTAD features,
dtype sanity, weight_bucket range, target shape.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.load_usaid import load_clean
from src.features.features import build_features, build_route_map, get_target, feature_names, LGBM_CAT_FEATURES


@pytest.fixture(scope="module")
def df():
    return load_clean()


@pytest.fixture(scope="module")
def X(df):
    return build_features(df, unctad_lookup=None)


@pytest.fixture(scope="module")
def y(df):
    return get_target(df)


def test_columns_present(X):
    missing = set(feature_names()) - set(X.columns)
    assert not missing, f"Missing columns: {missing}"


def test_shape_matches(df, X, y):
    assert len(X) == len(df)
    assert len(y) == len(df)


def test_no_nan_in_non_unctad_features(X):
    # route_label uses -1 for unknowns (not NaN), so it must also be complete
    non_unctad = [c for c in feature_names() if c != "unctad_rate_usd_kg"]
    for col in non_unctad:
        n_null = X[col].isnull().sum()
        assert n_null == 0, f"{col} has {n_null} NaN rows"


def test_unctad_all_nan_when_no_lookup(X):
    assert X["unctad_rate_usd_kg"].isnull().all()


def test_weight_bucket_range(X):
    vals = X["weight_bucket"].dropna()
    assert vals.min() >= 0
    assert vals.max() <= 4


def test_mode_encoded_range(X):
    assert X["mode_encoded"].between(0, 3).all()


def test_inco_encoded_range(X):
    assert X["inco_encoded"].between(0, 4).all()


def test_year_plausible(X):
    assert X["year"].between(2000, 2030).all()


def test_log_weight_positive(X):
    assert (X["log_weight"] >= 0).all()


def test_target_positive(y):
    assert (y > 0).all()


def test_cat_feature_names_in_feature_names():
    fn = feature_names()
    for cat in LGBM_CAT_FEATURES:
        assert cat in fn, f"{cat} not in feature_names()"


def test_route_label_unknown_returns_minus_one(df):
    """Unknown routes at inference must return -1, not raise.
    Simulate: build route_map on real training data, then apply to an unseen route."""
    train_route_map = build_route_map(df)
    inference_row = pd.DataFrame({
        "Weight_kg": [100.0],
        "mode": ["Air"],
        "inco_group": ["ExWorks"],
        "product_group": ["ARV"],
        "year": [2012],
        "origin_country": ["__unknown_origin__"],
        "dest_country": ["__unknown_dest__"],
    })
    X_infer = build_features(inference_row, unctad_lookup=None, route_map=train_route_map)
    assert X_infer["route_label"].iloc[0] == -1


def test_year_capped_at_max_train_year():
    import pandas as pd
    from src.features.features import build_features, MAX_TRAIN_YEAR
    dummy = pd.DataFrame({
        "Weight_kg": [100.0],
        "mode": ["Air"],
        "inco_group": ["ExWorks"],
        "product_group": ["ARV"],
        "year": [2026],
        "origin_country": ["India"],
        "dest_country": ["Kenya"],
    })
    X_dummy = build_features(dummy, unctad_lookup=None)
    assert X_dummy["year"].iloc[0] == MAX_TRAIN_YEAR
