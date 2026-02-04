from os.path import join, expanduser
import pandas as pd
import sys
import os
import shutil
import re
import warnings
from zrp.prepare.utils import load_file, load_json
from pathlib import Path
from zrp import ZRP
from zrp.prepare import ZGeo

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"

def test_import_working():
	import zrp

def test_load_data():
	csv_path = DATA_DIR / "sm_1.csv"
	df = pd.read_csv(csv_path)

	assert not df.empty

def test_initialize_zrp():
	zest_race_predictor = ZRP() 
	assert zest_race_predictor != None

def test_zrp_fit():
	zest_race_predictor = ZRP() 
	zest_race_predictor.fit()

def test_zrp_transform():
	csv_path = DATA_DIR / "sm_1.csv"
	df = pd.read_csv(csv_path)

	zest_race_predictor = ZRP() 
	zest_race_predictor.fit()
	print(df.columns)
	print(df['state'].apply(type))

	zrp_output = zest_race_predictor.transform(df)
	#print(zrp_output)


