---
title: Freight Cost Intelligence
emoji: 📦
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.57.0
app_file: app/streamlit_app.py
pinned: false
---

# Freight Cost Intelligence

Health commodity freight cost prediction on USAID shipment data, enriched with UNCTAD bilateral freight rate benchmarks as a lane-level cost signal. Includes a SHAP-grounded LLM explanation layer, a curated RAG corpus, and a what-if scenario chat that reruns the model on modified inputs.

## Problem

Predict USD freight cost for health commodity shipments given origin hub, destination country, mode (Air / Ocean / Truck), weight, commodity group, and INCO term. The output is a baseline cost estimate, the top SHAP feature drivers, a plain-English explanation grounded in those drivers, and retrieved trade document context.

The target use case is a supply chain analyst who needs a fast directional estimate and an explainable reason for the number — not a live market quote.

## Data

| Dataset | Role | Coverage |
|---|---|---|
| USAID SCMS Delivery History | Training | ~10,000 shipments, 2006–2015 |
| UNCTAD-World Bank GTCDIT | Enrichment feature | 170+ economies, bilateral $/kg rates, 2016–2021 |

**UNCTAD enrichment design:** For each training row, the UNCTAD median $/kg rate for (destination country, mode) is joined as `unctad_rate_usd_kg`. Fallback: direct match → mode-level global median → NaN. LightGBM handles NaN natively, so no imputation is applied.

**Temporal mismatch (known, disclosed):** USAID spans 2006–2015; UNCTAD covers 2016–2021. There is no year overlap. UNCTAD is used as a structural benchmark — bilateral freight rate intensities for a given corridor and mode are treated as stable structural signals even when drawn from a different period. This is the standard approach in gravity-model freight research. The UI and LIMITATIONS.md state this explicitly.

**Why not two separate models?** The originally planned second dataset (Brunel Shipping) was excluded after EDA revealed it was synthetic data. The UNCTAD enrichment replaced it as a structural signal rather than a second training source, which is a cleaner and more defensible architecture.

## Feature Set

| Feature | Description |
|---|---|
| `log_weight` | log1p(Weight_kg) |
| `weight_bucket` | Quantile bucket of log_weight (0–4) |
| `mode_encoded` | Air=0, Ocean=1, Truck=2, Other=3 |
| `inco_encoded` | ExWorks=0, DDP=1, CIF=2, RDC=3, Other=4 |
| `product_group_enc` | Label-encoded product group |
| `year` | Delivery year, capped at 2015 for inference |
| `route_label` | Dict-encoded (origin, destination) pair; -1 for unseen routes |
| `unctad_rate_usd_kg` | UNCTAD $/kg benchmark; NaN if no lane coverage |

## Model Results

| Metric | Linear Baseline | LightGBM |
|---|---|---|
| RMSE (log scale) | — | 0.807 |
| R² | — | 0.652 |

Time-based split: 80% train (pre-December 2013) / 20% test (post-December 2013). Random split is wrong for this data — the date column exists and the split must respect it.

`unctad_rate_usd_kg` appears in the top-8 SHAP features. The model is better on well-supported lanes (observed origin → destination → mode combinations in training). Unseen lane/mode combinations are blocked in the app before prediction.

## Pipeline Architecture

```
USAID CSV
    └─ load_usaid.py ──► cleaning, outlier capping, log target
UNCTAD CSV
    └─ load_unctad.py ──► lookup dict: (dest, mode) → $/kg

features.py ──► 8-feature row (UNCTAD join + encoding)
    └─ LightGBM (models/usaid_model.pkl)
        └─ prediction_usd + SHAP values

FAISS index (data/index/)
    └─ retriever.py ──► top-k chunks from corpus
        └─ generator.py ──► Groq/OpenAI LLM answer
        └─ explainer.py ──► deterministic SHAP prose (no LLM)
        └─ pipeline.py ──► rag_augmented_prediction()

whatif.py ──► intent parse → modify shipment → repredict → delta + explanation

streamlit_app.py ──► Tab 1: form + predict + explain
                     Tab 2: what-if chat
```

## RAG Corpus

Manually curated — quality over quantity. All chunks are traceable to source via YAML frontmatter.

| Directory | Content |
|---|---|
| `data/corpus/incoterms/` | One .md per Incoterm 2020 (EXW, FCA, FAS, FOB, CFR, CIF, CPT, CIP, DAP, DPU, DDP) |
| `data/corpus/modes/` | Air, ocean, road freight economics — cost drivers, USAID context |
| `data/corpus/glossary/` | 45-term shipping glossary |
| `data/corpus/hs_chapters/` | WCO HS chapter notes for health commodity categories (Ch. 30, 33, 38, 39, 40, 63, 85, 90) |
| `data/corpus/market_2026/` | Q1 2026 freight market context (IATA CTK, Drewry WCI, FBX) + Africa corridor benchmarks |

Sources: ICC Incoterms 2020, WCO HS 2017, IATA air cargo market analysis, Drewry World Container Index, Freightos Baltic Index, World Bank, AfDB corridor studies.

## What's Built

- [x] USAID EDA + cleaning pipeline + time-based train/test split
- [x] UNCTAD coverage audit (passed 30% gate), fallback hierarchy, feature join
- [x] Feature engineering + LightGBM model (RMSE 0.807 log / R² 0.652)
- [x] SHAP analysis — UNCTAD feature confirmed in top-8 drivers
- [x] RAG corpus (hand-curated, YAML-provenance, FAISS-indexed)
- [x] Retriever (FAISS, cosine similarity, threshold-filtered)
- [x] Generator (Groq primary / OpenAI fallback, LRU-cached)
- [x] Deterministic SHAP explainer (no LLM dependency for core explanation)
- [x] RAG pipeline orchestration
- [x] What-if chat (rule-based intent routing + model reprediction + delta)
- [x] Streamlit app — two tabs, input validation, support checks, session state
- [x] Test suite (pytest, 54+ passing)

Not yet done:
- [ ] HF Spaces deployment (model and FAISS index load fine locally; deployment pending)

## Running Locally

```bash
pip install -r requirements.txt
cp .env.example .env  # add GROQ_API_KEY
python src/rag/ingest.py  # build FAISS index (run once)
streamlit run app/streamlit_app.py
```

The app loads the saved model (`models/usaid_model.pkl`), FAISS index (`data/index/`), and UNCTAD lookup on startup. The prediction and deterministic explanation work without an API key. LLM-generated explanations require `GROQ_API_KEY`.

## Deployment on HuggingFace Spaces

This app is designed to run on HuggingFace Spaces with Streamlit SDK on free CPU hardware.

### Setup Steps

1. **Create a new HF Space:**
   - Owner: your HF org
   - SDK: Streamlit (v1.57.0)
   - Hardware: Free (CPU)
   - Visibility: Public or Protected

2. **Link to GitHub:**
   - Connect to the main branch of this repo
   - HF will auto-deploy on push

3. **Add API key to Secrets:**
   - In HF Space settings → "Secrets", add:
     - **GROQ_API_KEY**: your Groq API key (for LLM explanation & chat)

4. **Cold Start Note:**
   - **First load takes 1–2 minutes** (model, FAISS index, and UNCTAD lookup load into memory on app startup)
   - Subsequent loads are instant (assets stay cached)
   - CPU hardware is slower than GPU, but the app fits in memory easily

### Features

- **Tab 1 — Predict & Explain:**
  - Form: origin, destination, mode, weight, commodity, INCO, year
  - Output: LightGBM prediction, SHAP drivers, UNCTAD signal, RAG explanation + sources
  - "Try These Examples" buttons for instant demo shipments

- **Tab 2 — What-If Chat:**
  - Pre-seeded with shipment from Tab 1
  - Handles what-if questions ("What if I use sea instead of air?") and knowledge queries ("What does CIF mean?")
  - Shows modified prediction + delta + explanation + sources

- **"How It Works" Section:**
  - Collapsible diagram showing USAID → UNCTAD → Model → RAG → LLM pipeline
  - Links to limitations & design decisions

### Rate Limit Handling

If the Groq API hits rate limits (free tier has limits), the app shows:
> ⚠️ Groq API rate limit reached. Please try again in a moment. This is a temporary service limit, not an app error.

The prediction path works without the LLM, so basic predictions still succeed even if explanations fail.

## Tests

```bash
pytest tests/ -v
```

Tests mock LLM provider calls — no API key needed to run the suite.

## Stack

Python · pandas · scikit-learn · LightGBM · SHAP · sentence-transformers (all-MiniLM-L6-v2) · FAISS · Groq · Streamlit

## Data Ceiling

The main constraint on this project is data, not implementation. USAID SCMS data covers a narrow domain: HIV/ARV health commodities shipped to PEPFAR countries, 2006–2015, predominantly by air. A model trained on this distribution:

- Does not generalise to commercial freight
- Does not know post-2015 market conditions
- Cannot produce defensible predictions for lanes not in its training history

With access to a broader commercial freight dataset (carrier quotes, forwarding transactions, spot rate feeds), the same pipeline — the enrichment join, the SHAP-grounded explanation, the what-if chat — would produce substantially more useful predictions. The architecture is not the bottleneck.

See [LIMITATIONS.md](LIMITATIONS.md) for a full breakdown of what the system can and cannot do.

## Limitations

See [LIMITATIONS.md](LIMITATIONS.md).
