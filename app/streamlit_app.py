from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.load_unctad import get_rate, load_lookup
from src.data.load_usaid import INCO_MAP, load_clean
from src.rag.pipeline import available_market_context_years, rag_augmented_prediction
from src.rag.retriever import load_index
from src.rag.whatif import predict_shipment, whatif_chat

DEFAULT_SHIPMENT = {
    "origin_country": "India",
    "dest_country": "Kenya",
    "mode": "Air",
    "Weight_kg": 850.0,
    "inco_group": "ExWorks",
    "product_group": "ARV",
    "year": 2026,
}
STRUCTURAL_DISCLOSURE = (
    "USAID shipment data covers 2006-2015, while UNCTAD covers 2016-2021. "
    "The UNCTAD signal is shown here as a destination-tier structural benchmark rather than a time-matched join."
)
INCO_OPTIONS = ["ExWorks", "DDP", "CIF", "RDC", "Other"]
MODE_OPTIONS = ["Air", "Ocean", "Truck", "Other"]
BASELINE_YEAR = 2015

_ISO2_TO_COUNTRY = {
    "AF": "Afghanistan", "AL": "Albania", "DZ": "Algeria", "AO": "Angola",
    "AT": "Austria", "BE": "Belgium", "BJ": "Benin", "BW": "Botswana",
    "BR": "Brazil", "BF": "Burkina Faso", "BI": "Burundi", "CM": "Cameroon",
    "CA": "Canada", "CF": "Central African Republic", "TD": "Chad",
    "CN": "China", "CD": "Democratic Republic of Congo", "CG": "Congo",
    "CI": "Cote d'Ivoire", "CZ": "Czech Republic", "DK": "Denmark",
    "DJ": "Djibouti", "ER": "Eritrea", "SZ": "Eswatini", "ET": "Ethiopia",
    "FI": "Finland", "FR": "France", "GM": "Gambia", "DE": "Germany",
    "GH": "Ghana", "GN": "Guinea", "HT": "Haiti", "HU": "Hungary",
    "IN": "India", "ID": "Indonesia", "IE": "Ireland", "IT": "Italy",
    "JP": "Japan", "JO": "Jordan", "KE": "Kenya", "KR": "South Korea",
    "LS": "Lesotho", "LR": "Liberia", "MW": "Malawi", "ML": "Mali",
    "MR": "Mauritania", "MX": "Mexico", "MZ": "Mozambique", "NA": "Namibia",
    "NL": "Netherlands", "NE": "Niger", "NG": "Nigeria", "NO": "Norway",
    "PK": "Pakistan", "PL": "Poland", "PT": "Portugal", "RW": "Rwanda",
    "SN": "Senegal", "SL": "Sierra Leone", "SO": "Somalia", "ZA": "South Africa",
    "SS": "South Sudan", "ES": "Spain", "SD": "Sudan", "SE": "Sweden",
    "CH": "Switzerland", "TW": "Taiwan", "TZ": "Tanzania", "TH": "Thailand",
    "TG": "Togo", "UG": "Uganda", "GB": "United Kingdom", "US": "United States",
    "VN": "Vietnam", "ZM": "Zambia", "ZW": "Zimbabwe",
}
_COUNTRY_ALIASES: dict[str, str] = {
    "(USA)": "United States", "USA": "United States",
    "Korea": "South Korea", "DRC": "Democratic Republic of Congo",
    "Ivory Coast": "Cote d'Ivoire", "UK": "United Kingdom",
}
_COMPANY_SUFFIXES_UI = {
    "inc", "ltd", "llc", "lc", "llp", "corp", "co", "gmbh", "ag", "plc",
    "sa", "nv", "bv", "limited", "incorporated", "company", "pvt", "pte",
}
_COUNTRY_CONTINENT: dict[str, str] = {
    "United States": "Americas", "Canada": "Americas", "Mexico": "Americas",
    "Brazil": "Americas", "Haiti": "Americas",
    "France": "Europe", "Germany": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "United Kingdom": "Europe", "Switzerland": "Europe",
    "Sweden": "Europe", "Denmark": "Europe", "Austria": "Europe",
    "Spain": "Europe", "Poland": "Europe", "Czech Republic": "Europe",
    "Hungary": "Europe", "Portugal": "Europe", "Ireland": "Europe",
    "Finland": "Europe", "Norway": "Europe", "Italy": "Europe",
    "India": "Asia", "China": "Asia", "South Korea": "Asia", "Japan": "Asia",
    "Pakistan": "Asia", "Bangladesh": "Asia", "Vietnam": "Asia",
    "Thailand": "Asia", "Indonesia": "Asia", "Singapore": "Asia", "Taiwan": "Asia",
    "Jordan": "Asia",
    "Kenya": "Africa", "Uganda": "Africa", "Tanzania": "Africa",
    "Ethiopia": "Africa", "Nigeria": "Africa", "Ghana": "Africa",
    "South Africa": "Africa", "Zambia": "Africa", "Zimbabwe": "Africa",
    "Mozambique": "Africa", "Malawi": "Africa", "Rwanda": "Africa",
    "Cameroon": "Africa", "Senegal": "Africa", "Mali": "Africa",
    "Niger": "Africa", "Chad": "Africa", "Sudan": "Africa",
    "South Sudan": "Africa", "Democratic Republic of Congo": "Africa",
    "Congo": "Africa", "Cote d'Ivoire": "Africa", "Angola": "Africa",
    "Botswana": "Africa", "Namibia": "Africa", "Lesotho": "Africa",
    "Eswatini": "Africa", "Somalia": "Africa", "Eritrea": "Africa",
    "Djibouti": "Africa", "Burundi": "Africa", "Togo": "Africa",
    "Benin": "Africa", "Guinea": "Africa", "Sierra Leone": "Africa",
    "Liberia": "Africa", "Burkina Faso": "Africa", "Mauritania": "Africa",
    "Gambia": "Africa", "Central African Republic": "Africa",
    "Afghanistan": "Asia", "Algeria": "Africa", "Albania": "Europe",
}


def _normalize_origin_country(raw: str) -> str | None:
    """Map an extracted Manufacturing Site country to a clean display name.
    Returns None to signal the value should be filtered out."""
    raw = raw.strip()
    if not raw:
        return None
    if raw in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[raw]
    stripped = raw.strip("()")
    if stripped in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[stripped]
    if re.fullmatch(r"[A-Z]{2,3}", stripped):
        return _ISO2_TO_COUNTRY.get(stripped)
    if raw.lower() in _COMPANY_SUFFIXES_UI:
        return None
    return raw


def _truck_feasible(origin: str, dest: str) -> bool:
    """Return False if origin and destination are on different continents (truck impossible)."""
    o_cont = _COUNTRY_CONTINENT.get(origin)
    d_cont = _COUNTRY_CONTINENT.get(dest)
    if o_cont is None or d_cont is None:
        return True
    return o_cont == d_cont


@st.cache_data
def _load_reference_data() -> dict[str, Any]:
    df = load_clean()
    raw_origins = df["origin_country"].dropna().astype(str).unique()
    normalized = (_normalize_origin_country(v) for v in raw_origins if v)
    origins = sorted(v for v in normalized if v)
    destinations = sorted(value for value in df["dest_country"].dropna().astype(str).unique() if value)
    product_groups = sorted(value for value in df["product_group"].dropna().astype(str).unique() if value)
    route_mode_map = (
        df.groupby(["origin_country", "dest_country"])["mode"]
        .apply(lambda series: sorted(series.dropna().astype(str).unique().tolist()))
        .to_dict()
    )
    market_years = list(available_market_context_years())
    scenario_years = [BASELINE_YEAR, *[year for year in market_years if year != BASELINE_YEAR]]
    return {
        "df": df,
        "origins": origins,
        "destinations": destinations,
        "product_groups": product_groups,
        "scenario_years": scenario_years,
        "market_years": market_years,
        "route_mode_map": route_mode_map,
    }


@st.cache_resource
def _load_app_assets() -> dict[str, Any]:
    # Warm the actual cached repo assets on startup so the app doesn't invent a parallel stack.
    sample_prediction = predict_shipment(DEFAULT_SHIPMENT)
    unctad_lookup = load_lookup()
    index, chunks_df, _embed_model = load_index()
    return {
        "sample_prediction": sample_prediction,
        "unctad_lookup": unctad_lookup,
        "index_size": index.ntotal,
        "chunk_count": len(chunks_df),
    }


def _init_session_state() -> None:
    defaults = {
        "shipment": None,
        "last_prediction": None,
        "explanation_text": None,
        "unctad_signal": None,
        "context_year_used": None,
        "year_fallback_message": None,
        "chat_history": [],
        "open_chat_after_predict": False,
        "active_tab": "predict",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _index_for_value(options: list[str], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"${float(value):,.0f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.1f}%"


def _signal_label(rate: float | None) -> str:
    if rate is None or pd.isna(rate):
        return "No UNCTAD destination-tier signal available for this mode."
    return f"${float(rate):,.2f}/kg destination-tier structural benchmark"


def _lookup_unctad_signal(shipment: dict[str, Any], lookup: dict) -> float:
    return float(get_rate(lookup, shipment["dest_country"], shipment["mode"]))


def _build_shap_table(feature_names: list[str], shap_values: np.ndarray, feature_values: dict[str, Any], top_n: int = 6) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": np.asarray(shap_values, dtype=float),
            "feature_value": [feature_values.get(name) for name in feature_names],
        }
    )
    frame["abs_contribution"] = frame["contribution"].abs()
    frame = frame.sort_values("abs_contribution", ascending=False).head(top_n).copy()
    frame["direction"] = np.where(frame["contribution"] >= 0, "Up", "Down")
    frame["contribution"] = frame["contribution"].map(lambda x: round(float(x), 4))
    frame["feature_value"] = frame["feature_value"].map(_render_feature_value)
    return frame[["feature", "direction", "contribution", "feature_value"]]


def _render_feature_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "unknown"
        return f"{float(value):,.2f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _render_sources(sources: list[str]) -> None:
    if not sources:
        st.caption("No retrieved source list was returned for this response.")
        return
    st.caption("Sources")
    for source in sources:
        st.write(f"- {source}")


def _render_context_chunks(context_chunks: list[dict[str, Any]]) -> None:
    if not context_chunks:
        st.info("No retrieved market context was available for this shipment.")
        return
    for idx, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "Unknown source")
        category = chunk.get("category", "unknown")
        score = chunk.get("score")
        label = f"{idx}. {source} · {category}"
        if score is not None:
            label = f"{label} · score {float(score):.3f}"
        with st.expander(label):
            st.write(chunk.get("text", ""))


def _shipment_summary(shipment: dict[str, Any]) -> str:
    return (
        f"{shipment['origin_country']} -> {shipment['dest_country']} | "
        f"{shipment['mode']} | {shipment['Weight_kg']:,.0f} kg | "
        f"{shipment['product_group']} | {shipment['inco_group']} | scenario year {shipment['year']}"
    )


def _effective_model_year(year: int) -> int:
    return min(int(year), BASELINE_YEAR)


def _valid_modes_for_route(reference: dict[str, Any], origin_country: str, dest_country: str) -> list[str]:
    route_modes = reference["route_mode_map"].get((origin_country, dest_country))
    if route_modes:
        return route_modes
    return MODE_OPTIONS


def _route_mode_help_text(route_modes: list[str], selected_year: int) -> str:
    pieces = [
        "Model trained on 2006-2015 shipments.",
        f"Scenario years after {BASELINE_YEAR} are allowed, but the model year feature still uses a {BASELINE_YEAR}-equivalent baseline.",
        f"Effective model year for this input: {_effective_model_year(selected_year)}.",
    ]
    if route_modes != MODE_OPTIONS:
        pieces.append(f"Available modes for this route follow observed USAID lane history: {', '.join(route_modes)}.")
    else:
        pieces.append("This route was not observed directly in training, so all model modes remain available.")
    return " ".join(pieces)


def _build_validation_messages(reference: dict[str, Any], shipment: dict[str, Any]) -> list[str]:
    df = reference["df"]
    messages: list[str] = []

    origin = shipment["origin_country"]
    dest = shipment["dest_country"]
    mode = shipment["mode"]
    weight = float(shipment["Weight_kg"])
    year = int(shipment["year"])

    route_mask = (df["origin_country"] == origin) & (df["dest_country"] == dest)
    route_mode_mask = route_mask & (df["mode"] == mode)
    mode_mask = df["mode"] == mode

    route_count = int(route_mask.sum())
    route_mode_count = int(route_mode_mask.sum())
    mode_count = int(mode_mask.sum())

    if route_count == 0:
        messages.append(
            f"Unseen route: {origin} -> {dest} does not appear in the USAID training data, so the model is extrapolating."
        )
    elif route_mode_count == 0:
        observed_modes = sorted(df.loc[route_mask, "mode"].dropna().astype(str).unique().tolist())
        messages.append(
            f"Unseen route/mode combination: {origin} -> {dest} exists in training, but not with mode {mode}. "
            f"Observed modes for this lane: {', '.join(observed_modes)}."
        )

    if mode_count > 0:
        mode_weights = df.loc[mode_mask, "Weight_kg"].dropna().astype(float)
        p05 = float(mode_weights.quantile(0.05))
        p01 = float(mode_weights.quantile(0.01))
        if weight <= p01:
            messages.append(
                f"Very low-weight input for {mode}: {weight:,.1f} kg is at or below the 1st percentile of training support "
                f"({p01:,.1f} kg). Tiny parcel-style quotes are weakly supported by this model."
            )
        elif weight <= p05:
            messages.append(
                f"Low-weight input for {mode}: {weight:,.1f} kg is at or below the 5th percentile of training support "
                f"({p05:,.1f} kg). Predictions can flatten in this region."
            )

    if route_mode_count > 0:
        route_mode_weights = df.loc[route_mode_mask, "Weight_kg"].dropna().astype(float)
        min_seen = float(route_mode_weights.min())
        if weight < min_seen:
            messages.append(
                f"This lane/mode has no training examples below {min_seen:,.1f} kg. "
                f"Your input of {weight:,.1f} kg is outside observed support for {origin} -> {dest} via {mode}."
            )

    if year > BASELINE_YEAR:
        messages.append(
            f"Scenario year {year} still uses a {BASELINE_YEAR}-equivalent model year feature. "
            f"Only the retrieved market context reflects the later scenario year."
        )

    return messages


def _assess_prediction_support(reference: dict[str, Any], shipment: dict[str, Any]) -> dict[str, list[str]]:
    df = reference["df"]
    blockers: list[str] = []
    warnings: list[str] = []

    origin = shipment["origin_country"]
    dest = shipment["dest_country"]
    mode = shipment["mode"]
    weight = float(shipment["Weight_kg"])
    year = int(shipment["year"])

    route_mask = (df["origin_country"] == origin) & (df["dest_country"] == dest)
    route_mode_mask = route_mask & (df["mode"] == mode)
    mode_mask = df["mode"] == mode

    route_count = int(route_mask.sum())
    route_mode_count = int(route_mode_mask.sum())
    mode_count = int(mode_mask.sum())

    if year not in reference["scenario_years"]:
        blockers.append(
            f"Scenario year {year} is not supported in this build. Supported years: {', '.join(str(y) for y in reference['scenario_years'])}."
        )

    if mode == "Truck" and not _truck_feasible(origin, dest):
        blockers.append(
            f"Truck is not physically feasible for {origin} -> {dest}. Choose Air or Ocean for an intercontinental lane."
        )

    if route_count == 0:
        blockers.append(
            f"{origin} -> {dest} is an unseen route in the USAID training data. The numeric model is not defensible for this lane."
        )
    elif route_mode_count == 0:
        observed_modes = sorted(df.loc[route_mask, "mode"].dropna().astype(str).unique().tolist())
        blockers.append(
            f"{origin} -> {dest} via {mode} was not observed in training. Observed modes for this lane: {', '.join(observed_modes)}."
        )

    if mode_count > 0:
        mode_weights = df.loc[mode_mask, "Weight_kg"].dropna().astype(float)
        p01 = float(mode_weights.quantile(0.01))
        p05 = float(mode_weights.quantile(0.05))
        p95 = float(mode_weights.quantile(0.95))
        p99 = float(mode_weights.quantile(0.99))

        if weight < p01:
            blockers.append(
                f"{weight:,.1f} kg is below the 1st percentile of {mode} training support ({p01:,.1f} kg). "
                "Tiny parcel-style requests are out of scope for this model."
            )
        elif weight < p05:
            warnings.append(
                f"{weight:,.1f} kg is below the 5th percentile of {mode} training support ({p05:,.1f} kg). "
                "Predictions can flatten in this region."
            )

        if weight > p99:
            blockers.append(
                f"{weight:,.1f} kg is above the 99th percentile of {mode} training support ({p99:,.1f} kg). "
                "Extreme weights saturate the model and are blocked."
            )
        elif weight > p95:
            warnings.append(
                f"{weight:,.1f} kg is above the 95th percentile of {mode} training support ({p95:,.1f} kg). "
                "The model is extrapolating near the edge of observed weights."
            )

    if route_mode_count > 0:
        route_mode_weights = df.loc[route_mode_mask, "Weight_kg"].dropna().astype(float)
        min_seen = float(route_mode_weights.min())
        max_seen = float(route_mode_weights.max())
        if weight < min_seen:
            warnings.append(
                f"This lane/mode has no training examples below {min_seen:,.1f} kg. "
                f"Your input of {weight:,.1f} kg is outside observed support for {origin} -> {dest} via {mode}."
            )
        if weight > max_seen:
            warnings.append(
                f"This lane/mode has no training examples above {max_seen:,.1f} kg. "
                f"Your input of {weight:,.1f} kg is outside observed support for {origin} -> {dest} via {mode}."
            )

    if year > BASELINE_YEAR:
        warnings.append(
            f"Scenario year {year} still uses a {BASELINE_YEAR}-equivalent model year feature. "
            f"Only retrieved market context reflects the later scenario year."
        )

    if year not in reference["market_years"]:
        fallback_year = reference["market_years"][-1] if reference["market_years"] else None
        if fallback_year is not None:
            warnings.append(
                f"No year-specific market corpus exists for scenario year {year}. RAG context will fall back to {fallback_year}."
            )
        else:
            warnings.append(
                f"No year-specific market corpus exists for scenario year {year}, and no market context fallback is currently available."
            )

    return {"blockers": blockers, "warnings": warnings}


def _predict_and_explain(shipment: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
    prediction = predict_shipment(shipment)
    augmented = rag_augmented_prediction(
        prediction_usd=prediction["prediction_usd"],
        shap_values=prediction["shap_values"],
        feature_names=prediction["feature_names"],
        feature_values=prediction["feature_values"],
        shipment_mode=str(shipment["mode"]),
        destination_country=str(shipment["dest_country"]),
        scenario_year=shipment.get("year"),
        shipment_context=shipment,
    )
    signal = prediction["feature_values"].get("unctad_rate_usd_kg")
    return prediction, augmented, signal


def _handle_prediction_submit(shipment: dict[str, Any]) -> None:
    try:
        prediction, augmented, signal = _predict_and_explain(shipment)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.warning("Check that the model file, raw data, and local index assets are available.")
        return

    st.session_state["shipment"] = shipment
    st.session_state["last_prediction"] = {
        **prediction,
        "sources": augmented["sources"],
        "market_context": augmented["market_context"],
        "scenario_year_requested": augmented["scenario_year_requested"],
        "available_context_years": augmented["available_context_years"],
    }
    st.session_state["explanation_text"] = augmented["explanation"]
    st.session_state["unctad_signal"] = signal
    st.session_state["context_year_used"] = augmented["context_year_used"]
    st.session_state["year_fallback_message"] = augmented["year_fallback_message"]
    st.session_state["active_tab"] = "predict"


def _render_prediction_results() -> None:
    payload = st.session_state["last_prediction"]
    if not payload:
        return

    st.subheader("Prediction")
    requested_year = payload.get("scenario_year_requested")
    effective_year = _effective_model_year(requested_year or BASELINE_YEAR)
    st.metric("Model baseline prediction", _format_currency(payload["prediction_usd"]))
    st.caption(f"Scenario year: {requested_year} | effective model year feature: {effective_year}")

    signal = st.session_state["unctad_signal"]
    st.info(
        "UNCTAD destination-tier signal: "
        f"{_signal_label(signal)}"
    )
    st.caption(STRUCTURAL_DISCLOSURE)
    if st.session_state.get("year_fallback_message"):
        st.warning(st.session_state["year_fallback_message"])
    elif st.session_state.get("context_year_used") is not None:
        st.caption(f"Market context retrieval year: {st.session_state['context_year_used']}")

    explanation_text = st.session_state["explanation_text"]
    if explanation_text:
        st.subheader("AI Explanation")
        st.write(explanation_text)

    st.subheader("Top Feature Drivers")
    shap_table = _build_shap_table(
        feature_names=payload["feature_names"],
        shap_values=payload["shap_values"],
        feature_values=payload["feature_values"],
    )
    st.dataframe(shap_table, use_container_width=True, hide_index=True)

    st.subheader("Retrieved Context")
    _render_sources(payload.get("sources", []))
    _render_context_chunks(payload.get("market_context", []))

    if st.button("Open chat about this shipment", type="primary", use_container_width=True):
        st.session_state["open_chat_after_predict"] = True
        st.session_state["active_tab"] = "chat"
        st.success("Shipment is ready in What-If Chat. Open the second tab to continue.")

def _render_predict_tab(reference: dict[str, Any]) -> None:
    shipment = st.session_state["shipment"] or DEFAULT_SHIPMENT
    preselected_origin = shipment["origin_country"]
    preselected_dest = shipment["dest_country"]
    route_modes = _valid_modes_for_route(reference, preselected_origin, preselected_dest)
    selected_mode = shipment["mode"] if shipment["mode"] in route_modes else route_modes[0]

    with st.form("shipment_form"):
        col1, col2 = st.columns(2)
        with col1:
            origin_country = st.selectbox(
                "Origin country",
                options=reference["origins"],
                index=_index_for_value(reference["origins"], shipment["origin_country"]),
            )
            current_route_modes = _valid_modes_for_route(reference, origin_country, preselected_dest)
            mode = st.selectbox(
                "Mode",
                options=current_route_modes,
                index=_index_for_value(current_route_modes, selected_mode if selected_mode in current_route_modes else current_route_modes[0]),
                help="Modes are constrained by the route combinations observed in the USAID shipment data when available.",
            )
            inco_group = st.selectbox(
                "INCO group",
                options=INCO_OPTIONS,
                index=_index_for_value(INCO_OPTIONS, shipment["inco_group"]),
                help="Mapped repo grouping used by the trained model.",
            )
            year = st.selectbox(
                "Scenario year",
                options=reference["scenario_years"],
                index=_index_for_value(reference["scenario_years"], int(shipment["year"])),
                help="Only years with explicit app support are shown. The model still uses a 2015-equivalent baseline after 2015.",
            )
        with col2:
            dest_country = st.selectbox(
                "Destination country",
                options=reference["destinations"],
                index=_index_for_value(reference["destinations"], shipment["dest_country"]),
            )
            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=0.1,
                value=float(shipment["Weight_kg"]),
                step=50.0,
            )
            product_group = st.selectbox(
                "Product group",
                options=reference["product_groups"],
                index=_index_for_value(reference["product_groups"], shipment["product_group"]),
            )
            mapped_inco = st.text_input(
                "INCO term note",
                value="",
                placeholder="Optional note like EXW, DDP, CIF, or N/A - From RDC",
                help="Optional local note. The model uses the mapped INCO group above.",
            )

        submitted = st.form_submit_button("Predict & Explain", type="primary", use_container_width=True)

    current_route_modes = _valid_modes_for_route(reference, origin_country, dest_country)
    st.caption(_route_mode_help_text(current_route_modes, int(year)))

    if mapped_inco:
        normalized_term = INCO_MAP.get(mapped_inco.strip().upper())
        if normalized_term and normalized_term != inco_group:
            st.caption(f"Mapped note '{mapped_inco}' to repo group '{normalized_term}'.")

    shipment_candidate = {
        "origin_country": origin_country,
        "dest_country": dest_country,
        "mode": mode,
        "Weight_kg": float(weight_kg),
        "inco_group": inco_group,
        "product_group": product_group,
        "year": int(year),
    }
    assessment = _assess_prediction_support(reference, shipment_candidate)
    validation_messages = _build_validation_messages(reference, shipment_candidate)
    if assessment["blockers"] or assessment["warnings"] or validation_messages:
        st.subheader("Validation Checks")
        for message in assessment["blockers"]:
            st.error(message)
        for message in assessment["warnings"]:
            st.warning(message)
        for message in validation_messages:
            if message not in assessment["blockers"] and message not in assessment["warnings"]:
                st.info(message)

    if submitted:
        if assessment["blockers"]:
            st.error("Prediction blocked because this scenario is outside defensible model support.")
            return
        _handle_prediction_submit(shipment_candidate)

    _render_prediction_results()


def _render_chat_response(result: dict[str, Any]) -> None:
    if result["response_type"] == "rag_only":
        st.write(result.get("answer", "No answer returned."))
        _render_sources(result.get("sources", []))
        if result.get("context_chunks"):
            _render_context_chunks(result["context_chunks"])
        return

    st.metric("Modified prediction", _format_currency(result.get("modified_prediction_usd")))
    delta_label = _format_currency(result.get("delta_usd"))
    st.metric("Delta vs original", delta_label, _format_pct(result.get("delta_pct")))
    if result.get("year_fallback_message"):
        st.warning(result["year_fallback_message"])
    elif result.get("context_year_used") is not None:
        st.caption(f"Market context retrieval year: {result['context_year_used']}")
    st.write(result.get("explanation", "No explanation returned."))
    _render_sources(result.get("sources", []))
    if result.get("market_context"):
        _render_context_chunks(result["market_context"])


def _render_chat_tab() -> None:
    shipment = st.session_state["shipment"]
    if shipment is None:
        st.info("Use Predict & Explain first to create a shipment, then the what-if chat will use that live context.")
        return

    if st.session_state.get("open_chat_after_predict"):
        st.success("Using the shipment from Predict & Explain.")
        st.session_state["open_chat_after_predict"] = False

    top_left, top_right = st.columns([3, 2])
    with top_left:
        st.subheader("Current Shipment")
        st.write(_shipment_summary(shipment))
        signal = st.session_state["unctad_signal"]
        st.caption(f"UNCTAD destination-tier signal: {_signal_label(signal)}")
        st.caption(STRUCTURAL_DISCLOSURE)
        st.caption(
            f"Scenario year: {shipment['year']} | effective model year feature: {_effective_model_year(int(shipment['year']))}"
        )
        if st.session_state.get("year_fallback_message"):
            st.caption(st.session_state["year_fallback_message"])
    with top_right:
        if st.button("Reset chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
        if st.button("Reset shipment", use_container_width=True):
            st.session_state["shipment"] = None
            st.session_state["last_prediction"] = None
            st.session_state["explanation_text"] = None
            st.session_state["unctad_signal"] = None
            st.session_state["context_year_used"] = None
            st.session_state["year_fallback_message"] = None
            st.session_state["chat_history"] = []
            st.session_state["active_tab"] = "predict"
            st.rerun()

    st.caption("Try: What if I use sea instead of air?  |  What if destination is Uganda?  |  What does CIF mean?")

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and isinstance(message.get("content"), dict):
                _render_chat_response(message["content"])
            else:
                st.write(message["content"])

    user_message = st.chat_input("Ask a what-if question or a trade question")
    if not user_message:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    result = {"response_type": "rag_only", "answer": "", "sources": [], "context_chunks": []}
    with st.chat_message("assistant"):
        try:
            result = whatif_chat(user_message, shipment)
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Chat request failed: {exc}")
            result["answer"] = "Something went wrong. Check the terminal for details."
        _render_chat_response(result)

    st.session_state["chat_history"].append({"role": "assistant", "content": result})


def main() -> None:
    st.set_page_config(page_title="Freight Cost Intelligence", page_icon=":package:", layout="wide")
    _init_session_state()

    st.title("Freight Cost Intelligence")
    st.caption("Week 11 local Streamlit build using the current Week 8/9 pipeline.")

    reference = _load_reference_data()
    assets = _load_app_assets()

    with st.sidebar:
        st.subheader("Local Assets")
        st.write(f"Model warm-up sample: {_format_currency(assets['sample_prediction']['prediction_usd'])}")
        st.write(f"FAISS vectors: {assets['index_size']:,}")
        st.write(f"Chunk rows: {assets['chunk_count']:,}")
        st.caption("This app loads the saved model, local FAISS index, and UNCTAD lookup from the repo.")

    tabs = st.tabs(["Predict & Explain", "What-If Chat"])

    with tabs[0]:
        _render_predict_tab(reference)
    with tabs[1]:
        _render_chat_tab()


if __name__ == "__main__":
    main()
