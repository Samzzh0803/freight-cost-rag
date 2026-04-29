"""
Tests for UNCTAD lookup and fallback hierarchy.

Verifies: direct match, mode-level fallback, unknown mode → NaN,
UNCTAD enrichment coverage on real USAID rows, air > sea sanity check.
"""

import math
import pytest

from src.data.load_unctad import load_raw, build_lookup, get_rate
from src.data.load_usaid import load_clean
from src.features.features import build_features


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def lookup(raw_df):
    return build_lookup(raw_df)


def test_direct_match_returns_float(lookup):
    # Pick a real (dest, mode) pair that exists in the lookup
    real_key = next(k for k in lookup if isinstance(k, tuple))
    dest, mode = real_key
    rate = get_rate(lookup, dest, mode)
    assert isinstance(rate, float)
    assert rate > 0


def test_unknown_destination_falls_back_to_mode_median(lookup):
    rate = get_rate(lookup, "__nonexistent_country__", "Air")
    assert not math.isnan(rate), "Should fall back to mode median, not NaN"
    assert rate > 0


def test_unknown_mode_returns_nan(lookup):
    rate = get_rate(lookup, "__nonexistent_country__", "Other")
    assert math.isnan(rate), "Mode 'Other' has no UNCTAD data — should be NaN"


def test_known_destination_unknown_mode_returns_nan(lookup, raw_df):
    real_dest = raw_df["destination"].iloc[0]
    rate = get_rate(lookup, real_dest, "Other")
    assert math.isnan(rate)


def test_coverage_on_usaid_rows():
    """At least 30% of USAID rows should get a non-NaN UNCTAD rate."""
    from src.data.load_unctad import load_lookup
    df = load_clean()
    lookup = load_lookup()
    X = build_features(df, unctad_lookup=lookup)
    coverage = X["unctad_rate_usd_kg"].notna().mean()
    assert coverage >= 0.30, f"UNCTAD coverage {coverage:.1%} is below 30% threshold"


def test_air_rate_higher_than_sea_for_same_destination(lookup, raw_df):
    """Air freight should be more expensive per kg than sea freight."""
    # Find destinations that have both Air and Ocean rates
    air_keys = {k[0] for k in lookup if isinstance(k, tuple) and k[1] == "Air"}
    sea_keys = {k[0] for k in lookup if isinstance(k, tuple) and k[1] == "Ocean"}
    common = air_keys & sea_keys
    if not common:
        pytest.skip("No destination with both Air and Ocean rates")
    dest = next(iter(common))
    air_rate = get_rate(lookup, dest, "Air")
    sea_rate = get_rate(lookup, dest, "Ocean")
    assert air_rate > sea_rate, (
        f"Air ({air_rate:.4f}) should exceed Ocean ({sea_rate:.4f}) for {dest}"
    )


def test_lookup_has_mode_fallback_key(lookup):
    assert "__mode_fallback__" in lookup
    fallback = lookup["__mode_fallback__"]
    assert "Air" in fallback
    assert "Ocean" in fallback
