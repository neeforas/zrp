from zrp.zrp import ZRP
import pandas as pd
import numpy as np
from pathlib import Path

def test_load_data():
	file_path = Path(__file__).parent / "data" / "temp.parquet"
	df = pd.read_parquet(file_path)

	assert not df.empty

def test_initialize_zrp():
	zest_race_predictor = ZRP() 
	assert zest_race_predictor != None

def test_zrp_transform():
	# data cleaning
	mapping = {
    "FIRST_NAME": "first_name", 
    "MIDDLE_NAME": "middle_name", 
    "LAST_NAME": "last_name", 
    "HOUSE_NUMBER": "house_number", 
    "STREET_ADDRESS": "street_address", 
    "CITY": "city", 
    "STATE": "state", 
    "ZIP_CODE": "zip_code",
    "RECID": "ZEST_KEY",
	}

	data_path = Path(__file__).parent / "data" / "temp.parquet"
	df = pd.read_parquet(data_path)

	df["ADDRESS"] = (
		df["ADDRESS"]
		.astype(str)
		.str.replace(r"[^\x00-\x7F]+", "", regex=True)
	)

	df["address_split"] = df["ADDRESS"].str.split(" ")

	df["HOUSE_NUMBER"] = (
		df["address_split"]
		.str[0]
		.str.strip()
	)

	# keep only numeric house numbers
	df["HOUSE_NUMBER"] = (
		pd.to_numeric(df["HOUSE_NUMBER"], errors="coerce")
		.astype("Int64")
		.astype("string")
	)

	df["STREET_ADDRESS"] = np.where(
		df["HOUSE_NUMBER"].isna(),
		df["ADDRESS"],
		df["address_split"].str[1:].str.join(" ")
	)

	df["ZIP_CODE"] = df["ZIP_CODE"].astype("string")
	df["RECID"] = df["RECID"].astype("string")

	for c in ("CITY", "STATE", "ZIP_CODE", "HOUSE_NUMBER", "STREET_ADDRESS"):
		df[c] = df[c].replace("", pd.NA)
		
	df = df.rename(columns=mapping)

	df = df[list(mapping.values())]

	mask = ~(
		df["house_number"].isna()
		& df["city"].isna()
		& df["state"].isna()
		& (
			df["street_address"].isna()
			| (df["street_address"] == "CONFIDENTIAL")
		)
	)

	df = df.loc[mask].reset_index(drop=True)

	# clear artifacts folder
	folder = Path('./artifacts')
	try:
		for file in folder.iterdir():
			file.unlink()
	except OSError:
		pass


	zest_race_predictor = ZRP() 
	zrp_output = zest_race_predictor.transform(df)

	output_path = Path(__file__).parent / "geo_changes_results.csv"
	zrp_output.to_csv(output_path)

def test_compare_ct_results():
    cols = [
        "ZEST_KEY", "input_address", "match_status", "match_type",
        "matched_address", "coordinates", "tiger_line_id", "side",
        "state_fips", "county_fips", "tract", "block",
    ]

    df_census = pd.read_csv(
        Path(__file__).parent / "census_batch_results.csv",
        header=None, names=cols, dtype=str
    )
    df_census["GEOID_CT"] = df_census["state_fips"] + df_census["county_fips"] + df_census["tract"]

    zrp_results = pd.read_parquet(
        Path(__file__).parent / "artifacts" / "Zest_Geocoded_temp__2019__37_1.parquet"
    ).reset_index()

    merged = zrp_results[["ZEST_KEY", "GEOID_CT"]].astype(str).merge(
        df_census[["ZEST_KEY", "GEOID_CT", "state_fips", "county_fips", "tract"]],
        on="ZEST_KEY",
        suffixes=("_zrp", "_census")
    )

    merged = merged[
        (merged["GEOID_CT_zrp"] != "nan") & (merged["GEOID_CT_census"] != "nan")
    ]

    merged["CT_match"]        = merged["GEOID_CT_zrp"] == merged["GEOID_CT_census"]
    merged["ct_state_match"]  = merged["GEOID_CT_zrp"].str[:2]  == merged["state_fips"]
    merged["ct_county_match"] = merged["GEOID_CT_zrp"].str[2:5] == merged["county_fips"]
    merged["ct_tract_match"]  = merged["GEOID_CT_zrp"].str[5:]  == merged["tract"]

    print(f"\nMatched {len(merged)} records")
    print("\nFull CT match rate:\n",  merged["CT_match"].value_counts(normalize=True))
    print("\nState match rate:\n",    merged["ct_state_match"].value_counts(normalize=True))
    print("\nCounty match rate:\n",   merged["ct_county_match"].value_counts(normalize=True))
    print("\nTract match rate:\n",    merged["ct_tract_match"].value_counts(normalize=True))