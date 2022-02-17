import mne
import csv
import pandas as pd


file_dir = "C:/Users/Bruger/Documents/GitHub/edf/01_tcp_ar/002/00000297/s004_2012_02_28/00000297_s004_t000.edf"

file = mne.io.read_raw_edf(file_dir, preload=True)

data = file.filter(0.1, 100)

df = data.to_data_frame()


columns = [col for col in df.columns]
print(columns)
