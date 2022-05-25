import pickle
import csv
import re
import torch
import math

def series_dict_creator(path: str, encoding: str, name: str, num: int):
    """
    This function produce a series dictionary used in the training and
    validation of the networks and models.

    Args:
        path (str): path to where the data is stored.
        encoding (str): name of encoding file - needs to be in same dir as path.
        name (str): the name of the dictionary saved. Remeber to have .pickle at
                    the end of the name.
        num (int): the total number of recorded sessions at the path.

    Produced file:
        Series dictionary at the given path with the name provided.

        The file is index after the recordings (from 1 to num).
        For each recording the following meta data is saved in the file
        (in a list):
            Index 0: Total amount of samples from the patient that delivered
                     the recording.
            Index 1: How many times the patient has been recorded.
            Index 2: How many samples should be taken from the recording.
            Index 3: The size of the given recording. Given by
                     (#channels, #samples).
    """

    patient_samples = dict()
    # this dict will contain what patient delivered which recordings in the data

    # using information from the encoding file:
    with open(path + '/' + encoding, 'r') as file:
        ls = csv.reader(file)
        for rows in ls:
            m = re.match(r".+/\d+/(\d+)/+.", rows[0])
            val_to_dict = patient_samples.get(m.group(1), [])
            val_to_dict.append(rows[2])
            patient_samples[m.group(1)] = val_to_dict


    loader = [] # load each recording
    for i in range(1, num + 1):
        loader.append(torch.load(path + f"/model_input ({i}).pt"))

    file_size = dict() # get the dimention of each recording
    j = 0
    for file in loader:
        j += 1
        file_size[j] = file.shape


    ret_dict = dict() # The dictionary that is returned

    for values in patient_samples.values(): # for each patient
        vals = 0
        for i in values: # for the recordings the patient delivered
            vals += file_size[int(i)][1]
            # get the total number of samples recorded for the patient

        for i in values: # for the recordings the patient delivered
            length = math.floor(((file_size[int(i)][1] - 30*200)/(200*60*5))) \
                                            *file_size[int(i)][0]
            # this equalises the distrubtion of sampled intervals per patient

            res = int(min(torch.div(110, len(values), rounding_mode='trunc'),
                          length))
            # the res is to make sure that short series is not over represented

            ret_dict[int(i)] = [vals, len(values), res, file_size[int(i)]]



    # save the dictinary
    with open(path + "/" + name, 'wb') as handle:
        pickle.dump(ret_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    path = "C:/Users/Marc/Desktop/model_data/train_model_data"
    enc = "train_encoding.csv"
    name = "train_series_length.pickle"
    series_dict_creator(path, enc, name, 195) # create the dict for training

    path = "C:/Users/Marc/Desktop/model_data/val_model_data"
    enc = "val_encoding.csv"
    name = "val_series_length.pickle"
    series_dict_creator(path, enc, name, 28) # create the dict for training
