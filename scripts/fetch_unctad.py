"""
Fetch UNCTAD transport cost data via the UNCTADstat API and save to CSV.

Endpoint: US.TransportCosts — per-unit freight rate (USD/kg) by destination, mode, year.
Origin is fixed to US (code 0000), product to TOTAL. We filter to PEPFAR-relevant countries
using their M49 codes plus regional aggregates as fallback context.

Run once: python scripts/fetch_unctad.py
Output: data/raw/unctad/unctad_transport_costs_bilateral.csv
"""

import gzip
import io
import os
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Credentials — set env vars or hardcode temporarily for local dev
# ---------------------------------------------------------------------------
CLIENT_ID = os.environ.get("UNCTAD_CLIENT_ID", "6e58e1b2-00b9-42b8-b506-63017279a8a4")
CLIENT_SECRET = os.environ.get("UNCTAD_API_KEY", "8QqYCIFar3OAZN1jQYDxPCejnN14NeBBEkDBYn030Ng=")

API_URL = "https://unctadstat-user-api.unctad.org/US.TransportCosts/cur/Facts?culture=en"

# ---------------------------------------------------------------------------
# Filter parameters
# ---------------------------------------------------------------------------
# M49 codes for countries that appear in USAID data (PEPFAR destinations)
# plus a few regional aggregates as fallback context
DESTINATION_CODES = (
    # African PEPFAR countries
    "'566','800','894','404','716','834','646','231','508','710',"   # Nigeria,Uganda,Zambia,Kenya,Zimbabwe,Tanzania,Rwanda,Ethiopia,Mozambique,SouthAfrica
    "'180','516','072','706','120','288','466','024','694','454',"   # DRC,Namibia,Botswana,Somalia,Cameroon,Ghana,Mali,Angola,SierraLeone,Malawi
    "'108','729','148','686','862','686','768','204','466','226',"   # Burundi,Sudan,Chad,Senegal,Venezuela,Senegal,Togo,Benin,Mali,EquatorialGuinea
    "'710','768','288','204',"                                        # additional
    # Americas
    "'332','214','328',"                                              # Haiti,DominicanRepublic,Guyana
    # Asia
    "'704','586','050',"                                              # Vietnam,Pakistan,Bangladesh
    # Regional aggregates for fallback
    "'0001','5100','5200','5300','5400','5500','5600'"               # World,Africa,Americas,Europe,Asia,Oceania,Other
)

YEARS = "2016,2017,2018,2019,2020,2021"

# No TransportMode filter → returns all modes; we add TransportMode to $select
FILTER = (
    f"Origin/Code eq '0000' and Product/Code eq 'TOTAL' "
    f"and Destination/Code in ({DESTINATION_CODES}) "
    f"and Year in ({YEARS})"
)

SELECT = (
    "TransportMode/Label, Destination/Label, Year, "
    "Perunit_freight_rate_USkg_Value, Perunit_freight_rate_USkg_MissingValue"
)

COMPUTE = (
    "round(M1980/Value div 1, 3) as Perunit_freight_rate_USkg_Value, "
    "M1980/MissingValue/Label as Perunit_freight_rate_USkg_MissingValue"
)

ORDERBY = "TransportMode/Order asc, Destination/Order asc, Year asc"


def fetch_unctad() -> pd.DataFrame:
    payload = {
        "$select": SELECT,
        "$filter": FILTER,
        "$orderby": ORDERBY,
        "$compute": COMPUTE,
        "$format": "csv",
        "compress": "gz",
    }
    headers = {
        "ClientId": CLIENT_ID,
        "ClientSecret": CLIENT_SECRET,
    }

    print("Requesting UNCTAD transport cost data...")
    resp = requests.post(API_URL, data=payload, headers=headers, timeout=120)
    resp.raise_for_status()

    print(f"Response status: {resp.status_code}, size: {len(resp.content):,} bytes")

    with gzip.open(io.BytesIO(resp.content)) as f:
        df = pd.read_csv(f, encoding="utf-8", na_values=[""])

    print(f"Rows fetched: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTransport modes found:\n{df.iloc[:, 0].value_counts()}")
    return df


if __name__ == "__main__":
    df = fetch_unctad()

    out_path = "data/raw/unctad/unctad_transport_costs_bilateral.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(df.head(10).to_string())
