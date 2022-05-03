# This is a script to generate the memeory map for input and target data.

import numpy as np
import pickle
from tempfile import mkdtemp
import torch

def mem_creator(path: str, num: int):

    input_ls = []
    target_ls = []
    for i in range(1, num + 1):
        input_ls.append(torch.load(path + f"/model_input ({i}).pt"))
        target_ls.append(torch.load(path + f"/model_target ({i}).pt"))

    most_chan = 0
    longest = 0

    for val in input_ls:
        if val.shape[0] > most_chan:
            most_chan = val.shape[0]

        if val.shape[1] > longest:
            longest = val.shape[1]

    print("Most channels:", most_chan)
    print("Longest series:", longest)


    ind_file = path + "/model_input.dat"
    tar_file = path + "/model_target.dat"
    #filename = path.join(mkdtemp(), 'newfile.dat')
    ind = np.memmap(ind_file, dtype='float32', mode='w+',
                    shape=(len(input_ls), most_chan, longest))

    tar = np.memmap(tar_file, dtype='float32', mode='w+',
                    shape=(len(input_ls), most_chan, longest))

    for i in range(0, num):
        s = input_ls[i].shape
        ind[i, 0:s[0], 0:s[1]] = input_ls[i]
        tar[i, 0:s[0], 0:s[1]] = target_ls[i]




if __name__ == "__main__":
    path = "C:/Users/Marc/Desktop/model_data/train_model_data"
    mem_creator(path, 195) # create the dict for training

    path = "C:/Users/Marc/Desktop/model_data/val_model_data"
    name = "val_series_length.pickle"
    mem_creator(path, 28) # create the dict for training
