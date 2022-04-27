import pickle
import csv
import re
import torch
import math

def series_dict_creator(path: str, encoding: str, name: str, num: int):

    patient_samples = dict()

    with open(path + '/' + encoding, 'r') as file:
        ls = csv.reader(file)
        for rows in ls:
            m = re.match(r".+/\d+/(\d+)/+.", rows[0])
            val_to_dict = patient_samples.get(m.group(1), [])
            val_to_dict.append(rows[2])
            patient_samples[m.group(1)] = val_to_dict


    loader = []
    for i in range(1, num + 1):
        loader.append(torch.load(path + f"/model_input ({i}).pt"))

    file_size = dict()
    j = 0
    for file in loader:
        j += 1
        file_size[j] = file.shape


    ret_dict = dict() # The dictionary that is returned

    for values in patient_samples.values():
        vals = 0
        for i in values:
            vals += file_size[int(i)][1]

        # for i in values:
        #     length = math.floor(((file_size[int(i)][1]-30*200)/(200*60*5))) \
        #                                     *file_size[int(i)][0]
        #
        #     res = int(min(torch.div(110, len(values), rounding_mode='trunc'),
        #                   length))
        #
        #     ret_dict[int(i)] = res

        for i in values:      #REMEMBER MINUS

            length = math.floor(((file_size[int(i)][1] - 30*200)/(200*60*5))) \
                                            *file_size[int(i)][0]

            res = int(min(torch.div(110, len(values), rounding_mode='trunc'),
                          length))

            ret_dict[int(i)] = [vals, len(values), res, file_size[int(i)]]

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
