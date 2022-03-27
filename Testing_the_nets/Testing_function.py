import neptune.new as neptune
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CyclicLR
import torch.multiprocessing as mp
import numpy as np
import random

import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Unet import Unet
from LoaderPACK.Loader import load_whole_data, load_5_min_intervals, load_shuffle_5_min
from LoaderPACK.Accuarcy_finder import Accuarcy_find_tester
from multiprocessing import Process
from LoaderPACK.trainer import net_train
import matplotlib.pyplot as plt
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Set up the datasets
random.seed(42)



#val_set, train_set = torch.utils.data.random_split(
#                            random.sample(range(1, 226 + 1), 200), [26, 200],
#                            generator=torch.Generator().manual_seed(42))

val_set, train_set = torch.utils.data.random_split(
                            random.sample(range(1, 226 + 1), 50), [10, 40],
                            generator=torch.Generator().manual_seed(42))


set = [i for i in val_set]

load_file = load_whole_data(path = "C:/Users/Marc/Desktop/model_data",
                                  ind = val_set)

file_loader = torch.utils.data.DataLoader(load_file,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0)

batch_size = 1

i = 1

valid_acc = torch.tensor([]).to(device)

model = Unet(n_channels = 1, n_classes = 2)
x = torch.load(f"C:/Users/Marc/Desktop/network/SGD_net{i}.pt")
model.load_state_dict(x)
model.to(device)

j = 0

for file in file_loader:
    if False:
        figure, axis = plt.subplots(8, 1)

        # The artifacts on channel: FP1-F7
        axis[0].plot(file[0][0][0])
        axis[0].axes.xaxis.set_visible(False) # remove digits on x-axis

        # The artifacts on channel: F7-T3
        axis[1].plot(file[1][0][0], "b")
        axis[1].axes.xaxis.set_visible(False) # remove digits on x-axis

        axis[2].plot(file[0][0][1])
        axis[2].axes.xaxis.set_visible(False) # remove digits on x-axis

        # The artifacts on channel: F7-T3
        axis[3].plot(file[1][0][1], "b")
        axis[3].axes.xaxis.set_visible(False) # remove digits on x-axis


        axis[4].plot(file[0][0][2])
        axis[4].axes.xaxis.set_visible(False) # remove digits on x-axis

        # The artifacts on channel: F7-T3
        axis[5].plot(file[1][0][2], "b")
        axis[5].axes.xaxis.set_visible(False) # remove digits on x-axis

        axis[6].plot(file[0][0][3])
        axis[6].axes.xaxis.set_visible(False) # remove digits on x-axis

        # The artifacts on channel: F7-T3
        axis[7].plot(file[1][0][3], "b")
        axis[7].axes.xaxis.set_visible(False) # remove digits on x-axis

        plt.show()


    load_series = load_5_min_intervals(file, device)

    series_loader = torch.utils.data.DataLoader(load_series,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)
    v_mat = torch.zeros(2,2)
    total_pos, total_neg = torch.tensor(0), torch.tensor(0)

    for series in series_loader:
        ind, tar, chan, cut = series
        y_pred = model(ind)
        #pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
        #target = tar.view(-1).type(it)

        acc, mat, tot_p_g, tot_n_g, art_pred = Accuarcy_find_tester(y_pred, tar, device)
        valid_acc = torch.cat((valid_acc, acc.view(1)))
        v_mat = v_mat + mat
        total_pos = total_pos + tot_p_g
        total_neg = total_neg + tot_n_g

        #with open("C:/Users/Marc/Desktop/info/data_encoding.csv", "w", newline='') as f:
        #    write = csv.writer(f) # save information that link the nr of the
        #                          # .pt files with the .edf files.
        #    write.writerow(1*art_pred[0].cpu())

        if False:
            figure, axis = plt.subplots(2, 1)

            # The artifacts on channel: FP1-F7
            axis[0].plot(ind[0][0].cpu())
            axis[0].axes.xaxis.set_visible(False) # remove digits on x-axis

            # The artifacts on channel: F7-T3
            axis[1].plot(1*art_pred[0].cpu(), "b")
            axis[1].plot(tar[0][0].cpu(), "r", markersize=1)
            axis[1].set_title(f"Nr:{set[j]} with channel: {chan} at {cut}")
            axis[1].axes.xaxis.set_visible(False) # remove digits on x-axis

            plt.show()
            input("stop")

    j += 1

plt.plot(valid_acc.cpu())
plt.show()
