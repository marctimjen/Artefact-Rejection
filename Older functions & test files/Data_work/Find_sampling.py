import csv
import mne
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import numpy as np

# Was created to find the different sampling frequencies.


def make_file_list(edf_list: str, csv_list: str, data_dir: str) -> list:
    file_list = []
    file1 = open(edf_list)
    file2 = open(csv_list)
    reader1 = csv.reader(file1) # reader for the edf file locations
    reader2 = csv.reader(file2) # reader for the rec file location

    for i in zip(reader1, reader2):
        first = data_dir + i[0][0][2:]
        second = data_dir + i[1][0][2:-3] + 'csv'
        file_list.append([first, second])

    file1.close()
    file2.close()
    return file_list # returns list of file locations


dir1_edf_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/edf_01_tcp_ar.list"
dir2_edf_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/edf_02_tcp_le.list"
dir3_edf_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/edf_03_tcp_ar_a.list"

dir1_csv_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/rec_01_tcp_ar.list"
dir2_csv_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/rec_02_tcp_le.list"
dir3_csv_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/rec_03_tcp_ar_a.list"


if __name__ == "__main__":
    dir_edf_list = [dir1_edf_list, dir2_edf_list, dir3_edf_list]
    dir_csv_list = [dir1_csv_list, dir2_csv_list, dir3_csv_list]

    data_dir = "C:/Users/Marc/Desktop/data/v2.1.0"
    nr = 1

    sampling_set = set()

    for i in range(0, 3):
        dir_edf = dir_edf_list[i]
        dir_rec = dir_csv_list[i]

        file_list = make_file_list(dir_edf, dir_rec, data_dir)

        for i in file_list:
            edf_dir = i[0]
            data = mne.io.read_raw_edf(edf_dir, preload=False)
            data = data.resample(sfreq=200)
            sfreq = int(data.info['sfreq'])
            sampling_set.add(sfreq)

    print(sampling_set)
