# Freight Cost Intelligence — Project A1

Health commodity freight cost prediction on USAID SCMS shipment data, enriched with
UNCTAD bilateral freight rate benchmarks as a lane-level cost signal.

## Problem

Predict USD freight cost for health commodity shipments (Air / Truck / Ocean) given
origin hub, destination country, mode, weight, commodity, and INCO term.
Target audience: supply chain analysts at PEPFAR-funded programs who need fast cost
estimates and plain-English explanations of what drives them.

## Data

| Dataset | Role | Records |
|---|---|---|
| USAID SCMS Delivery History (2006–2015) | Training data | ~10,000 shipments |
| UNCTAD-World Bank GTCDIT (2016–2021) | Lookup enrichment | 170+ economy bilateral rates |

**UNCTAD enrichment approach:** For each USAID shipment, we attach the UNCTAD median
per-unit freight rate (USD/kg) for that destination country and mode as a lane-level
benchmark feature. Fallback hierarchy: direct `(destination, mode)` match → mode-level
global median → NaN (LightGBM handles NaN natively). The UNCTAD CSV does not include
HS-code granularity at the bilateral lane level, so HS4→HS2 fallback from the original
plan was collapsed into the mode-level median.

**Temporal mismatch (known, intentional):** USAID spans 2006–2015; UNCTAD covers
2016–2021. No year overlap. UNCTAD is used as a structural benchmark — bilateral freight
rate intensities for a given lane and mode are sufficiently stable structural signals
even when drawn from a different period. This is standard practice in gravity-model
freight research. Stated explicitly here and in interviews.

**Ad-valorem note:** v4 project plan references `unctad_adval_rate` (%). The UNCTAD
bulk CSV downloaded for this project (`US.TransportCosts`) only exposes the per-unit
rate (`Perunit_freight_rate_USkg_Value`). The feature is `unctad_rate_usd_kg` ($/kg),
not a percentage. Interview framing updated accordingly.

## Feature Set

| Feature | Type | Description |
|---|---|---|
| `log_weight` | continuous | log1p(Weight_kg) |
| `weight_bucket` | int 0–4 | quantile bucket of log_weight |
| `mode_encoded` | int 0–3 | Air=0, Ocean=1, Truck=2, Other=3 |
| `inco_encoded` | int 0–4 | ExWorks=0, DDP=1, CIF=2, RDC=3, Other=4 |
| `product_group_enc` | int | label-encoded product group |
| `year` | int | delivery year, capped at 2015 for inference |
| `route_label` | int | dict-encoded (origin, dest) pair; -1 for unknown routes |
| `unctad_rate_usd_kg` | float | UNCTAD $/kg benchmark; NaN if no coverage |

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · sentence-transformers ·
FAISS · Groq · Streamlit · Hugging Face Spaces

## Results

_To be filled after Week 4 (baseline + tree models)._

## Running Tests

```bash
pytest tests/ -v
```

## Project Status

- [x] Week 1: USAID EDA
- [x] Week 2: UNCTAD EDA + coverage audit (passed 30% gate)
- [x] Week 3: Feature engineering, tests, time-based split
- [ ] Week 4: Baseline + tree models + SHAP stub
- [ ] Weeks 5–13: UNCTAD analysis, RAG, what-if chat, Streamlit, deploy
