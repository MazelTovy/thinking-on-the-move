#!/usr/bin/env python3
"""
Download ACS 2019 5-Year demographic data at Census Block Group level
for New York State (FIPS 36). Same variables as DTBK project.

Requires: pip install census us
API Key: f66cc9cc400512e95e7633b75daca2c24b1ac487

Output: data/acs_cbg_demographics.csv
"""

import os
import pandas as pd
from census import Census

API_KEY = "f66cc9cc400512e95e7633b75daca2c24b1ac487"
YEAR = 2019
OUT_PATH = "/scratch/sx2490/econai/nyc_metro/data/acs_cbg_demographics.csv"

# ACS 5-Year variables
VARIABLES = {
    "B01001_001E": "total_population",
    "B01002_001E": "median_age",
    "B19013_001E": "median_household_income",
    "B19301_001E": "per_capita_income",
    "B02001_002E": "white_population",
    "B02001_003E": "black_population",
    "B02001_005E": "asian_population",
    "B03002_012E": "hispanic_population",
    "B15003_022E": "bachelor_degree",
    "B15003_023E": "master_degree",
    "B15003_024E": "professional_degree",
    "B15003_025E": "phd_degree",
    "B23025_002E": "labor_force",
    "B23025_004E": "employed",
    "B23025_005E": "unemployed",
    "B25001_001E": "housing_units",
    "B25002_002E": "occupied_units",
    "B25002_003E": "vacant_units",
    "B25003_002E": "owner_occupied",
    "B25003_003E": "renter_occupied",
    "B25077_001E": "median_home_value",
    "B25064_001E": "median_gross_rent",
    "B11001_001E": "households",
}

# States to download: NY (36) and NJ (34) for commuters
STATES = {"36": "NY", "34": "NJ"}


def main():
    c = Census(API_KEY, year=YEAR)
    all_data = []

    for state_fips, state_name in STATES.items():
        print(f"Downloading {state_name} (FIPS {state_fips})...")
        var_list = list(VARIABLES.keys())

        data = c.acs5.state_county_blockgroup(
            fields=["NAME"] + var_list,
            state_fips=state_fips,
            county_fips="*",
            tract="*",
            blockgroup="*"
        )
        print(f"  Retrieved {len(data):,} block groups")
        all_data.extend(data)

    df = pd.DataFrame(all_data)

    # Rename columns
    df = df.rename(columns=VARIABLES)

    # Build standard block_group_id (12-digit GEOID)
    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["tract"] = df["tract"].astype(str).str.zfill(6)
    df["block group"] = df["block group"].astype(str)
    df["block_group_id"] = df["state"] + df["county"] + df["tract"] + df["block group"]

    # Also create Cuebiq-format ID (US.NY.XXX.TTTTTT.B)
    state_map = {"36": "NY", "34": "NJ"}
    df["cbg_cuebiq"] = df.apply(
        lambda r: f"US.{state_map.get(r['state'], r['state'])}.{r['county']}.{r['tract']}.{r['block group']}",
        axis=1
    )

    # Convert to numeric
    numeric_cols = list(VARIABLES.values())
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Census uses negative values as error codes
        df.loc[df[col] < 0, col] = None

    # Derive shares
    pop = df["total_population"].replace(0, None)
    df["white_share"] = (df["white_population"] / pop).clip(0, 1)
    df["black_share"] = (df["black_population"] / pop).clip(0, 1)
    df["asian_share"] = (df["asian_population"] / pop).clip(0, 1)
    df["hispanic_share"] = (df["hispanic_population"] / pop).clip(0, 1)

    hu = df["housing_units"].replace(0, None)
    df["rent_share"] = (df["renter_occupied"] / hu).clip(0, 1)

    lf = df["labor_force"].replace(0, None)
    df["employment_rate"] = (df["employed"] / lf).clip(0, 1)
    df["unemployment_rate"] = (df["unemployed"] / lf).clip(0, 1)

    # Drop intermediate columns
    df = df.drop(columns=["NAME", "state", "county", "tract", "block group"], errors="ignore")

    # Reorder
    id_cols = ["block_group_id", "cbg_cuebiq"]
    other_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + other_cols]

    # Remove duplicates
    df = df.drop_duplicates(subset=["block_group_id"])

    print(f"\nTotal block groups: {len(df):,}")
    print(f"  NY: {(df['block_group_id'].str[:2] == '36').sum():,}")
    print(f"  NJ: {(df['block_group_id'].str[:2] == '34').sum():,}")

    # Missing data summary
    print(f"\nMissing data:")
    for col in numeric_cols + ["white_share", "black_share", "asian_share",
                                "hispanic_share", "rent_share", "employment_rate"]:
        if col in df.columns:
            nmiss = df[col].isna().sum()
            if nmiss > 0:
                print(f"  {col}: {nmiss} ({nmiss/len(df)*100:.1f}%)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
