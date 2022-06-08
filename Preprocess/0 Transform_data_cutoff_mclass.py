import csv
import mne
import pandas as pd
import torch
import math
import numpy as np
import re


def make_file_list(edf_list: str, csv_list: str, data_dir: str) -> list:
    """
    This function is used to make two lists with directories to the edf files
    and to the files with the target annotations.

    Args:
        edf_list (str): path to a file containing directories to the edf files.
        csv_list (str): path to a file containing directories to the csv files.
        data_dir (str): first part of the path.

    Return:
        file_list: list with paths to the edf files and target annotation files.
    """
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
    """
    This function creates .pt (torch files) for the input (eeg-recordings) and
    target (annotaion) data.

    Args:
        file_list (list): list with paths to the edf and annotation files.
        montage (dict): montage for derivation used.
        save_loc (str): directory in which the data is saved.

    Produced files:
        This function creates input and target files from the edf and annotaion
        data. The file format is .pt used in pytorch.
    """
    global nr
    for direct in file_list:
        edf_dir = direct[0]
        csv_dir = direct[1]

        data = mne.io.read_raw_edf(edf_dir, preload=True) # read edf file

        data = data.filter(0.1, 100) # use filter on data
        data = data.notch_filter(60) # use filter on data

        data = data.resample(sfreq=200) # resample the data into 200

        if data.__len__() < 60*5.5*200: # if the file has less than 5,5 mins of
            continue                    # recorded data then it's discarded

        df = data.to_data_frame() # make pandas dataframe

        inv_map = {v[0]: k for k, v in montage.items()}
        # to make sure, that the correct targets are given to the right
        # channels, the index order is used.

        which_montages = set()
        target = []
        with open(csv_dir, "r") as file: # read rec file
            ls = csv.reader(file)        # open the annotaion file
            skip = 0
            flag = True
            for rows in ls:
                if rows[0][0] == "#": # if the row starts with "#" then move on
                    continue
                elif flag:            # jump over the header (name of columns)
                    flag = False
                    continue


                elec_flag = re.search("elec", rows[3].lower()) or re.search("elpp", rows[3].lower())

                if elec_flag: # test if string is in the elec class
                    tar_anno = 2 # normally = 4
                # elif rows[3].lower() == "musc":
                #     tar_anno = 3
                # elif rows[3].lower() == "eyem":
                #     tar_anno = 4 # was 2
                else:
                    tar_anno = 1

                # tar anno gets the correct annotation of the form:
                # class 4 (all with electrode artifacts)
                # class 3 only muscle artifacts ALONE
                # class 2 eye movement artifact alone
                # class 1 catch all (shiv and chew)
                # class 0 no artefact.

                target.append([inv_map.get(str(rows[0])), float(rows[1]),
                                float(rows[2]), tar_anno])

                which_montages.add(inv_map.get(str(rows[0])))


        sorted_index = sorted(list(which_montages)) # sort the montage index

        bad_idx = []
        first = True
        for i in sorted_index: # using the montage information we make the new
            col_names = montage.get(i) # data-frame using only the channels
                                       # that has been labeled

            if col_names[0] == "EKG": # & first # special case that is removed
                continue

            if first:
                list1 = df[col_names[1]] # get the first series
                list2 = df[col_names[2]] # get the second series
                df_new = list1 - list2
                df_new = pd.DataFrame(df_new.rename(col_names[0])) # Rename
                df_new.loc[df_new[col_names[0]] < -200] = -200
                df_new.loc[df_new[col_names[0]] > 200] = 200
                # cut the top and bot
                first = False
            else:
                try:
                    list1 = df[col_names[1]]
                    list2 = df[col_names[2]]
                except:
                    bad_idx.append(i)
                    target = [t for t in target if t[0] != i]
                    continue

                diff = list1 - list2
                diff = diff.rename(col_names[0]) # Rename

                diff.loc[diff < -200] = -200
                diff.loc[diff > 200] = 200
                df_new = pd.concat([df_new, diff], axis=1, join='inner')

        for i in bad_idx:
            sorted_index.remove(i)

        tar = torch.zeros(df_new.shape[1], df_new.shape[0]) # make target data

        for i in target: # i = [montage_channel, start, end, type_artifact]
            index = sorted_index.index(i[0])
            # Find the correct index in the target

            tar[index][200 * math.floor(i[1]): 200 * math.ceil(i[2])] = i[3]
                # Make the artifacts = 1

        ind = torch.tensor(df_new.values.T) # data-frame to tensor
        torch.save(ind, save_loc + f'/model_input ({nr}).pt') # save input
        torch.save(tar, save_loc + f'/model_target ({nr}).pt') # save target


        with open(save_loc + "/data_encoding.csv", "a", newline='') as f:
            write = csv.writer(f) # save information that link the nr of the
                                  # .pt files with the .edf files.
            write.writerow([edf_dir, csv_dir, nr, sorted_index])

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
    21: ["P4-O2", "EEG P4-REF", "EEG O2-REF"]
}
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

dir1_edf_list = "C:/Users/Marc/Desktop/data_series/v2.1.0/lists/edf_01_tcp_ar.list"
dir2_edf_list = "C:/Users/Marc/Desktop/data_series/v2.1.0/lists/edf_02_tcp_le.list"
dir3_edf_list = "C:/Users/Marc/Desktop/data_series/v2.1.0/lists/edf_03_tcp_ar_a.list"

dir1_csv_list = "C:/Users/Marc/Desktop/data_series/v2.1.0/lists/rec_01_tcp_ar.list"
dir2_csv_list = "C:/Users/Marc/Desktop/data_series/v2.1.0/lists/rec_02_tcp_le.list"
dir3_csv_list = "C:/Users/Marc/Desktop/data_series/v2.1.0/lists/rec_03_tcp_ar_a.list"


if __name__ == "__main__":
    montage_list = [montage1, montage2, montage1]
    # just use montage2 always - this has all channels
    dir_edf_list = [dir1_edf_list, dir2_edf_list, dir3_edf_list]
    dir_csv_list = [dir1_csv_list, dir2_csv_list, dir3_csv_list]

    data_dir = "C:/Users/Marc/Desktop/data_series/v2.1.0"
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
