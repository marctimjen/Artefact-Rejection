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
import matplotlib.pyplot as plt

import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Unet_leaky import Unet_leaky
from LoaderPACK.Loader import shuffle_5min
from LoaderPACK.Accuarcy_finder import Accuarcy_find
from LoaderPACK.Accuarcy_upload import Accuarcy_upload
from multiprocessing import Process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if device == "cpu":
    fl = torch.FloatTensor
    it = torch.LongTensor
else:
    fl = torch.cuda.FloatTensor
    it = torch.cuda.LongTensor

batch_size = 10

                                  #"/home/tyson/model_data/train_model_data"

train_load_file = shuffle_5min(path = "C:/Users/Marc/Desktop/model_data/train_model_data",
                                     series_dict = 'train_series_length.pickle',
                                     size = (195, 22, 2060000),
                                     device = device)


train_loader = torch.utils.data.DataLoader(train_load_file,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)

val_load_file = shuffle_5min(path = "C:/Users/Marc/Desktop/model_data/val_model_data",
                                     series_dict = 'val_series_length.pickle',
                                     size = (28, 22, 549200),
                                     device = device)


val_loader = torch.utils.data.DataLoader(val_load_file,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)

model = Unet_leaky(n_channels=1, n_classes=2).to(device)
optimizer = SGD(model.parameters(), lr=0.001)
lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device), reduction = "mean")

nEpoch = 3

train_loss = []
valid_loss = []


for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")

    # run[f"network_SGD/learning_rate"].log(optimizer.param_groups[0]['lr'])

    t_mat = torch.zeros(2, 2)
    total_pos, total_neg = torch.tensor(0), torch.tensor(0)


    for series in train_loader:
        ind, tar, chan = series
        y_pred = model(ind)
        model.zero_grad()
        pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
        target = tar.view(-1).type(it)
        loss = lossFunc(pred, target)

        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())


    for series in val_loader:
        ind, tar, chan = series
        y_pred = model(ind)
        pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
        target = tar.view(-1).type(it)
        loss = lossFunc(pred, target)
        valid_loss.append(loss.item())


plt.plot(valid_loss)
plt.show()
