import csv
import mne
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import numpy as np


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

# https://stackoverflow.com/questions/20910213/loop-over-two-generator-together


def read_and_export_files(file_list: list, montage: dict, save_loc: str):
    global nr
    for direct in file_list:
        edf_dir = direct[0]
        csv_dir = direct[1]

        data = mne.io.read_raw_edf(edf_dir, preload=True) # read edf file

        if data.__len__() < 82500: # if the file has less than 5,5 mins of
            continue               # recorded data then it's discarded

        data = data.filter(0.1, 100) # use filter on data
        data = data.notch_filter(60) # use filter on data

        sfreq = int(data.info['sfreq']) # get the sampling freqency

        if sfreq != 250:
            data = data.resample(sfreq=250)

        df = data.to_data_frame() # make pandas dataframe

        inv_map = {v[0]: k for k, v in montage1.items()}
        # to make sure, that the correct targets are given to the right
        # channels, the index order is used.

        which_montages = set()
        target = []
        with open(csv_dir, "r") as file: # read rec file
            ls = csv.reader(file)
            skip = 0
            for rows in ls:
                if rows[0][0] == "#":
                    continue
                target.append([inv_map.get(str(rows[1])), float(rows[2]),
                                float(rows[3])])
                which_montages.add(inv_map.get(str(rows[1])))

        sorted_index = sorted(list(which_montages)) # sort the montage index

        first = True
        for i in sorted_index: # using the montage information we make the new
            col_names = montage.get(i) # data-frame using only the channels
                                       # that has been labeled

            if (col_names[0] == "EKG"): # & first # special case that is removed
                continue
        #        df_new = df[col_names[1]]
        #        df_new = df_new.rename(col_names[0])
        #        first = False
        #    elif (col_names[0] == "EKG"): # special case for montage 2
        #        list1 = df[col_names[1]]
        #        list1 = list1.rename(col_names[0])
        #        df_new = pd.concat([df_new, diff], axis=1, join='inner')

            if first:
                list1 = df[col_names[1]] # get the first series
                list2 = df[col_names[2]] # get the second series
                df_new = list1 - list2
                df_new = pd.DataFrame(df_new.rename(col_names[0])) # Rename
                first = False
            else:
                list1 = df[col_names[1]]
                list2 = df[col_names[2]]
                diff = list1 - list2
                diff = diff.rename(col_names[0]) # Rename
                df_new = pd.concat([df_new, diff], axis=1, join='inner')


        tar = torch.zeros(df_new.shape[1], df_new.shape[0]) # make target data

        for i in target: # i = [montage_channel, start, end, type_artifact]
            index = sorted_index.index(i[0]) # Find the correct index in the target
            tar[index][250 * math.floor(i[1]): 250 * math.ceil(i[2])] = 1
                # Make the artifacts = 1

        ind = torch.tensor(df_new.values.T) # data-frame to tensor
        torch.save(ind, save_loc + f'/model_input ({nr}).pt') # save input
        torch.save(tar, save_loc + f'/model_target ({nr}).pt') # save target


        with open(save_loc + "/data_encoding.csv", "a", newline='') as f:
            write = csv.writer(f) # save information that link the nr of the
                                  # .pt files with the .edf files.
            write.writerow([edf_dir, csv_dir, nr])

        nr += 1

montage1 = {
0: ["FP1-F7", "EEG FP1-REF", "EEG F7-REF"],
1: ["F7-T3", "EEG F7-REF", "EEG T3-REF"],
2: ["T3-T5", "EEG T3-REF", "EEG T5-REF"],
3: ["T5-O1", "EEG T5-REF", "EEG O1-REF"],
4: ["FP2-F8", "EEG FP2-REF", "EEG F8-REF"],
5: ["F8-T4", "EEG F8-REF", "EEG T4-REF"],
6: ["T4-T6", "EEG T4-REF", "EEG T6-REF"],
7: ["T6-O2", "EEG T6-REF", "EEG O2-REF"],
8: ["A1-T3", "EEG A1-REF", "EEG T3-REF"],
9: ["T3-C3", "EEG T3-REF", "EEG C3-REF"],
10: ["C3-CZ", "EEG C3-REF", "EEG CZ-REF"],
11: ["CZ-C4", "EEG CZ-REF", "EEG C4-REF"],
12: ["C4-T4", "EEG C4-REF", "EEG T4-REF"],
13: ["T4-A2", "EEG T4-REF", "EEG A2-REF"],
14: ["FP1-F3", "EEG FP1-REF", "EEG F3-REF"],
15: ["F3-C3", "EEG F3-REF", "EEG C3-REF"],
16: ["C3-P3", "EEG C3-REF", "EEG P3-REF"],
17: ["P3-O1", "EEG P3-REF", "EEG O1-REF"],
18: ["FP2-F4", "EEG FP2-REF", "EEG F4-REF"],
19: ["F4-C4", "EEG F4-REF", "EEG C4-REF"],
20: ["C4-P4", "EEG C4-REF", "EEG P4-REF"],
21: ["P4-O2", "EEG P4-REF", "EEG O2-REF"]}
montage2 = {
0: ["FP1-F7", "EEG FP1-LE", "EEG F7-LE"],
1: ["F7-T3", "EEG F7-LE", "EEG T3-LE"],
2: ["T3-T5", "EEG T3-LE", "EEG T5-LE"],
3: ["T5-O1", "EEG T5-LE", "EEG O1-LE"],
4: ["FP2-F8", "EEG FP2-LE", "EEG F8-LE"],
5: ["F8-T4", "EEG F8-LE", "EEG T4-LE"],
6: ["T4-T6", "EEG T4-LE", "EEG T6-LE"],
7: ["T6-O2", "EEG T6-LE", "EEG O2-LE"],
8: ["A1-T3", "EEG A1-LE", "EEG T3-LE"],
9: ["T3-C3", "EEG T3-LE", "EEG C3-LE"],
10: ["C3-CZ", "EEG C3-LE", "EEG CZ-LE"],
11: ["CZ-C4", "EEG CZ-LE", "EEG C4-LE"],
12: ["C4-T4", "EEG C4-LE", "EEG T4-LE"],
13: ["T4-A2", "EEG T4-LE", "EEG A2-LE"],
14: ["FP1-F3", "EEG FP1-LE", "EEG F3-LE"],
15: ["F3-C3", "EEG F3-LE", "EEG C3-LE"],
16: ["C3-P3", "EEG C3-LE", "EEG P3-LE"],
17: ["P3-O1", "EEG P3-LE", "EEG O1-LE"],
18: ["FP2-F4", "EEG FP2-LE", "EEG F4-LE"],
19: ["F4-C4", "EEG F4-LE", "EEG C4-LE"],
20: ["C4-P4", "EEG C4-LE", "EEG P4-LE"],
21: ["P4-O2", "EEG P4-LE", "EEG O2-LE"],
22: ["EKG", "EEG EKG-LE", "None"]
}
montage3 = {
0: ["FP1-F7", "EEG FP1-REF", "EEG F7-REF"],
1: ["F7-T3", "EEG F7-REF", "EEG T3-REF"],
2: ["T3-T5", "EEG T3-REF", "EEG T5-REF"],
3: ["T5-O1", "EEG T5-REF", "EEG O1-REF"],
4: ["FP2-F8", "EEG FP2-REF", "EEG F8-REF"],
5: ["F8-T4", "EEG F8-REF", "EEG T4-REF"],
6: ["T4-T6", "EEG T4-REF", "EEG T6-REF"],
7: ["T6-O2", "EEG T6-REF", "EEG O2-REF"],
8: ["T3-C3", "EEG T3-REF", "EEG C3-REF"],
9: ["C3-CZ", "EEG C3-REF", "EEG CZ-REF"],
10: ["CZ-C4", "EEG CZ-REF", "EEG C4-REF"],
11: ["C4-T4", "EEG C4-REF", "EEG T4-REF"],
12: ["FP1-F3", "EEG FP1-REF", "EEG F3-REF"],
13: ["F3-C3", "EEG F3-REF", "EEG C3-REF"],
14: ["C3-P3", "EEG C3-REF", "EEG P3-REF"],
15: ["P3-O1", "EEG P3-REF", "EEG O1-REF"],
16: ["FP2-F4", "EEG FP2-REF", "EEG F4-REF"],
17: ["F4-C4", "EEG F4-REF", "EEG C4-REF"],
18: ["C4-P4", "EEG C4-REF", "EEG P4-REF"],
19: ["P4-O2", "EEG P4-REF", "EEG O2-REF"]
}

dir1_edf_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/edf_01_tcp_ar.list"
dir2_edf_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/edf_02_tcp_le.list"
dir3_edf_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/edf_03_tcp_ar_a.list"

dir1_csv_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/rec_01_tcp_ar.list"
dir2_csv_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/rec_02_tcp_le.list"
dir3_csv_list = "C:/Users/Marc/Desktop/data/v2.1.0/lists/rec_03_tcp_ar_a.list"


if __name__ == "__main__":
    montage_list = [montage1, montage2, montage3]
    dir_edf_list = [dir1_edf_list, dir2_edf_list, dir3_edf_list]
    dir_csv_list = [dir1_csv_list, dir2_csv_list, dir3_csv_list]

    data_dir = "C:/Users/Marc/Desktop/data/v2.1.0"
    nr = 1

    save_loc = "C:/Users/Marc/Desktop/model_data" # location to save files
    f = open(save_loc + "/data_encoding.csv", 'w')
    f.close() # file for the encoding

    for i in range(0, 3):
        dir_edf = dir_edf_list[i]
        dir_rec = dir_csv_list[i]

        file_list = make_file_list(dir_edf, dir_rec, data_dir)

        montage = montage_list[i] # find the correct montage
        read_and_export_files(file_list, montage, save_loc)
