from zrp.zrp import ZRP
import pandas as pd
import numpy as np
from pathlib import Path

def test_load_data():
	file_path = Path(__file__).parent / "data" / "nc.parquet"
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

	data_path = Path(__file__).parent / "data" / "nc.parquet"
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

	zrp_results_path = Path(__file__).parent / "artifacts" / "Zest_Geocoded__2019__37_1.parquet"
	zrp_results = pd.read_parquet(zrp_results_path)
	assert not zrp_results['GEOID_CT'].empty

def test_compare_ct_results():
	# reads batch results to a dataframe
	cols = [
		"ZEST_KEY",
		"input_address",
		"match_status",
		"match_type",
		"matched_address",
		"coordinates",
		"tiger_line_id",
		"side",
		"state_fips",
		"county_fips",
		"tract",
		"block",
	]

	batch_results_path = Path(__file__).parent / "census_batch_results.csv"
	df_census = pd.read_csv(
		batch_results_path,
		header=None,
		names=cols,
		dtype=str
	)

	df_census["GEOID_CT"] = (
		df_census["state_fips"]
		+ df_census["county_fips"]
		+ df_census["tract"]
	)

	df_census["GEOID_BG"] = (
		df_census["GEOID_CT"]
		+ df_census["block"].str[0]
	)

	zrp_results_path = Path(__file__).parent / "artifacts" / "Zest_Geocoded__2019__37_1.parquet"
	zrp_results = pd.read_parquet("artifacts/Zest_Geocoded__2019__37_1.parquet")
	merged = zrp_results.merge(
    df_census[["ZEST_KEY", "GEOID_CT", "GEOID_BG", "state_fips", "county_fips", "tract"]],
    on="ZEST_KEY",
    suffixes=("_zrp", "_census")
	)

	# drop failed entries
	merged = merged.dropna(subset=["GEOID_CT_zrp", "GEOID_CT_census"])

	merged["CT_match"] = merged["GEOID_CT_zrp"] == merged["GEOID_CT_census"]
	print("CT Id match: ")
	print(merged["CT_match"].value_counts(normalize=True))

	merged["ct_state_match"] = merged["state_fips"] == merged["GEOID_CT_zrp"].str[:2]
	merged["ct_county_match"] = merged["county_fips"] == merged["GEOID_CT_zrp"].str[2:5]
	merged["ct_tract_match"] = merged["tract"] == merged["GEOID_CT_zrp"].str[5:]

	print("States match: ")
	print(merged["ct_state_match"].value_counts(normalize=True))
	print("Counties match match: ")
	print(merged["ct_county_match"].value_counts(normalize=True))
	print("Tracts match: ")
	print(merged["ct_tract_match"].value_counts(normalize=True))

	print(merged.columns)
	print(merged["CT_match"].value_counts(normalize=True))

	merged.groupby("ct_state_match")["ct_county_match"].value_counts(normalize=True)
	



