# Made by Juliano E. S. Padua
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils import load_config

paths, config = load_config()

# initialize paths from config
data_raw_path = paths["data_raw"]
data_processed_path = paths["data_processed"]
images_path = paths["images"]
report_path = paths["report"]
addons_path = paths["addons"]

# your code starts here
current_datetime = datetime.datetime.now()

# Project 'FinData Cleaner' initialized on 2025-09-26 15:18:45
