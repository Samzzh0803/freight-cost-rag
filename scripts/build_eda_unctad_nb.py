"""One-shot script to write 02_eda_unctad.ipynb."""
import json

cells = [
    {
        "cell_type": "markdown",
        "id": "md-title",
        "metadata": {},
        "source": (
            "# 02 — UNCTAD EDA\n\n"
            "**Goal:** Schema tour, coverage audit against USAID lanes, rate sanity check, "
            "correlation, go/no-go decision on UNCTAD feature.\n\n"
            "## Findings\n_Fill in after running all cells._\n\n"
            "- Coverage: X% direct match, Y% mode-fallback, Z% NaN\n"
            "- Sanity: Air > Ocean > Truck rates? YES / NO\n"
            "- Correlation (UNCTAD rate vs USAID cost at lane level): r = ?\n"
            "- **Decision gate verdict: GO / NO-GO**"
        ),
    },
    {
        "cell_type": "code",
        "id": "imports",
        "metadata": {},
        "source": "\n".join([
            "import sys",
            'sys.path.insert(0, "..")',
            "",
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from scipy import stats",
            "",
            "from src.data.load_usaid import load_raw as load_usaid_raw, clean as clean_usaid",
            "from src.data.load_unctad import load_raw as load_unctad_raw, load_lookup, get_rate",
            "",
            'sns.set_theme(style="whitegrid")',
            'pd.set_option("display.max_columns", 20)',
        ]),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "md-schema",
        "metadata": {},
        "source": "## 1 — Schema Tour\n\nWhat columns did the UNCTAD API give us?",
    },
    {
        "cell_type": "code",
        "id": "schema",
        "metadata": {},
        "source": "\n".join([
            "unctad = load_unctad_raw()",
            'print("Shape:", unctad.shape)',
            'print("\\nDtypes:")',
            "print(unctad.dtypes)",
            'print("\\nYear range:", unctad["year"].min(), "to", unctad["year"].max())',
            'print("\\nModes:", unctad["mode"].unique().tolist())',
            'print("\\nUnique destinations:", unctad["destination"].nunique())',
            "unctad.head(8)",
        ]),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "md-usaid",
        "metadata": {},
        "source": (
            "## 2 — Load USAID (cleaned)\n\n"
            "We need `dest_country` and `mode` columns to attempt the UNCTAD join."
        ),
    },
    {
        "cell_type": "code",
        "id": "load-usaid",
        "metadata": {},
        "source": "\n".join([
            "usaid = clean_usaid(load_usaid_raw())",
            'print("USAID shape:", usaid.shape)',
            'print("USAID modes:", usaid["mode"].value_counts().to_dict())',
            'print("\\nSample dest_country values:")',
            'print(usaid["dest_country"].value_counts().head(10))',
        ]),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "md-coverage",
        "metadata": {},
        "source": (
            "## 3 — Coverage Audit\n\n"
            "For every USAID row, attempt UNCTAD lookup on `(dest_country, mode)`.\n\n"
            "- **Direct hit**: destination + mode found in lookup dict\n"
            "- **Mode fallback**: destination not found, used mode-level median\n"
            "- **NaN**: mode itself not in UNCTAD (should be 0)\n\n"
            "Threshold: if <30% non-NaN after fallback → drop UNCTAD feature."
        ),
    },
    {
        "cell_type": "code",
        "id": "coverage",
        "metadata": {},
        "source": "\n".join([
            "lookup = load_lookup()",
            "",
            "# Keys that are real (dest, mode) pairs",
            'direct_keys = {k for k in lookup if k != "__mode_fallback__"}',
            "",
            "def classify_lookup(dest, mode):",
            "    if (dest, mode) in direct_keys:",
            '        return "direct"',
            '    if lookup.get("__mode_fallback__", {}).get(mode) is not None:',
            '        return "mode_fallback"',
            '    return "nan"',
            "",
            'usaid["lookup_class"] = usaid.apply(',
            '    lambda r: classify_lookup(r["dest_country"], r["mode"]), axis=1',
            ")",
            'usaid["unctad_rate"] = usaid.apply(',
            '    lambda r: get_rate(lookup, r["dest_country"], r["mode"]), axis=1',
            ")",
            "",
            'counts = usaid["lookup_class"].value_counts()',
            "pct = (counts / len(usaid) * 100).round(1)",
            'print("Coverage breakdown:")',
            'for cls in ["direct", "mode_fallback", "nan"]:',
            "    n = counts.get(cls, 0)",
            "    p = pct.get(cls, 0.0)",
            '    print(f"  {cls:15s}: {n:5d} rows ({p}%)")',
            "non_nan_pct = pct.get('direct', 0.0) + pct.get('mode_fallback', 0.0)",
            "print(f\"\\nNon-NaN total: {usaid['unctad_rate'].notna().sum()} rows ({non_nan_pct:.1f}%)\")",
        ]),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "md-sanity",
        "metadata": {},
        "source": (
            "## 4 — Sanity Check: Air > Ocean > Truck?\n\n"
            "Air freight should be most expensive per kg. "
            "If this ordering is wrong, mode mapping has a problem."
        ),
    },
    {
        "cell_type": "code",
        "id": "sanity",
        "metadata": {},
        "source": "\n".join([
            'mode_rates = unctad.groupby("mode")["rate_usd_per_kg"].median().sort_values(ascending=False)',
            'print("Median UNCTAD $/kg by mode:")',
            "print(mode_rates.round(3))",
            "",
            "fig, ax = plt.subplots(figsize=(6, 3.5))",
            'mode_rates.plot(kind="bar", ax=ax, color=["steelblue","seagreen","tomato","goldenrod"])',
            'ax.set_title("UNCTAD Median $/kg by Mode")',
            'ax.set_ylabel("$/kg")',
            'ax.set_xlabel("")',
            "plt.xticks(rotation=0)",
            "plt.tight_layout()",
            "plt.show()",
            "",
            'air = mode_rates.get("Air", 0)',
            'ocean = mode_rates.get("Ocean", 0)',
            'truck = mode_rates.get("Truck", 0)',
            'print(f"\\nAir ({air:.3f}) > Ocean ({ocean:.3f})? {air > ocean}")',
            'print(f"Ocean ({ocean:.3f}) > Truck ({truck:.3f})? {ocean > truck}")',
        ]),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "md-dist",
        "metadata": {},
        "source": "## 5 — Rate Distribution for Matched USAID Rows",
    },
    {
        "cell_type": "code",
        "id": "dist",
        "metadata": {},
        "source": "\n".join([
            'matched = usaid[usaid["unctad_rate"].notna()].copy()',
            "",
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))",
            "",
            'axes[0].hist(matched["unctad_rate"], bins=40, color="steelblue", edgecolor="white")',
            'axes[0].set_title("UNCTAD Rate Distribution (USAID rows)")',
            'axes[0].set_xlabel("$/kg")',
            'axes[0].set_ylabel("count")',
            "",
            'for mode, grp in matched.groupby("mode"):',
            '    axes[1].hist(grp["unctad_rate"], bins=30, alpha=0.6, label=mode)',
            'axes[1].set_title("UNCTAD Rate by Mode")',
            'axes[1].set_xlabel("$/kg")',
            "axes[1].legend()",
            "",
            "plt.tight_layout()",
            "plt.show()",
            "",
            'print(matched.groupby("mode")["unctad_rate"].describe().round(3))',
        ]),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "md-corr",
        "metadata": {},
        "source": (
            "## 6 — Correlation: UNCTAD Rate vs USAID Actual Cost\n\n"
            "Aggregate to lane level (dest_country + mode). "
            "Compare median UNCTAD rate vs median USAID freight cost per lane.\n\n"
            "This is the **go/no-go signal**."
        ),
    },
    {
        "cell_type": "code",
        "id": "correlation",
        "metadata": {},
        "source": "\n".join([
            "lane = (",
            "    matched",
            '    .groupby(["dest_country", "mode"])',
            "    .agg(",
            '        median_usaid_cost=("Freight_Cost_USD", "median"),',
            '        median_unctad_rate=("unctad_rate", "median"),',
            '        n_shipments=("Freight_Cost_USD", "count"),',
            "    )",
            "    .reset_index()",
            ")",
            'print(f"Unique lanes: {len(lane)}")',
            "",
            'r, p = stats.pearsonr(lane["median_unctad_rate"], np.log1p(lane["median_usaid_cost"]))',
            'print(f"Pearson r (UNCTAD rate vs log USAID cost): {r:.3f}  p={p:.4f}")',
            "",
            "fig, ax = plt.subplots(figsize=(7, 5))",
            "sc = ax.scatter(",
            '    lane["median_unctad_rate"],',
            '    np.log1p(lane["median_usaid_cost"]),',
            '    c=lane["n_shipments"], cmap="viridis", alpha=0.7, s=60',
            ")",
            'plt.colorbar(sc, ax=ax, label="shipment count")',
            "",
            'xv = lane["median_unctad_rate"]',
            'yv = np.log1p(lane["median_usaid_cost"])',
            "m, b = np.polyfit(xv, yv, 1)",
            "x_line = np.linspace(xv.min(), xv.max(), 100)",
            'ax.plot(x_line, m*x_line+b, "r--", lw=1.5, label=f"r={r:.2f}")',
            "",
            'ax.set_xlabel("Median UNCTAD $/kg")',
            'ax.set_ylabel("log(Median USAID Freight Cost USD)")',
            'ax.set_title("Lane-Level: UNCTAD Benchmark vs USAID Actual Cost")',
            "ax.legend()",
            "plt.tight_layout()",
            "plt.show()",
        ]),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "md-gate",
        "metadata": {},
        "source": "## 7 — Decision Gate",
    },
    {
        "cell_type": "code",
        "id": "gate",
        "metadata": {},
        "source": "\n".join([
            "direct_pct = pct.get('direct', 0.0)",
            "fallback_pct = pct.get('mode_fallback', 0.0)",
            "nan_pct = pct.get('nan', 0.0)",
            "non_nan_pct = direct_pct + fallback_pct",
            "",
            'print("=" * 50)',
            'print("WEEK 2 DECISION GATE")',
            'print("=" * 50)',
            'print(f"Coverage (non-NaN):  {non_nan_pct:.1f}%  (threshold: >30%)")',
            'print(f"  Direct match:      {direct_pct:.1f}%")',
            'print(f"  Mode fallback:     {fallback_pct:.1f}%")',
            'print(f"  NaN:               {nan_pct:.1f}%")',
            'print(f"Correlation r:       {r:.3f}  (meaningful if |r| > 0.1)")',
            'print()',
            "",
            "if non_nan_pct >= 30 and abs(r) > 0.1:",
            '    print("VERDICT: GO — proceed to Week 3 feature engineering.")',
            "elif non_nan_pct < 30:",
            '    print("VERDICT: NO-GO — coverage below 30%. Drop UNCTAD feature.")',
            "else:",
            '    print("VERDICT: NO-GO — correlation near zero. UNCTAD adds no signal.")',
        ]),
        "outputs": [],
        "execution_count": None,
    },
    {
        "cell_type": "markdown",
        "id": "md-findings",
        "metadata": {},
        "source": (
            "## Findings\n\n"
            "_Fill in after running all cells._\n\n"
            "- **Coverage**: X% direct match, Y% mode-fallback, Z% NaN\n"
            "- **Sanity check**: Air > Ocean > Truck? YES / NO\n"
            "- **Correlation r**: ?\n"
            "- **Decision gate verdict**: GO / NO-GO\n\n"
            "### Limitations\n"
            "- USAID 2006–2015 vs UNCTAD 2016–2021: no year overlap. "
            "Used as structural benchmark, not time-matched.\n"
            "- UNCTAD data is US import aggregated across origins — join is on "
            "(destination, mode) only, not full bilateral pairs.\n"
            "- Mode fallback rows get the global mode median, not a lane-specific rate."
        ),
    },
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open("notebooks/02_eda_unctad.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Written successfully.")
