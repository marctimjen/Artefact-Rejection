import csv
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.LSTM_net import LSTM_net
from LoaderPACK.naive_models import thenos, linear_model
from LoaderPACK.Loader import testload_5min
from LoaderPACK.tester import val_tester
from LoaderPACK.Unet import Unet
from LoaderPACK.Accuarcy_finder import Accuarcy_find
from LoaderPACK.Accuarcy_upload import Accuarcy_upload
from multiprocessing import Process
from LoaderPACK.Unet_leaky import Unet_leaky, Unet_leaky_lstm

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


# "/home/tyson/data_cutoff/val_model_data"
# val_load_file = testload_5min(path = "C:/Users/Marc/Desktop/data_cutoff/val_model_data",
#                                      series_dict = 'val_series_length.pickle',
#                                      size = (28, 22, 549200),
#                                      device = "cpu")
#
#
# val_loader = torch.utils.data.DataLoader(val_load_file,
#                                                 batch_size=1,
#                                                 shuffle=False,
#                                                 num_workers=0)
# j = 0
# for i in val_loader:
#     print(i)
#     j += 1
#
# print("Total series loaded:", j)





batch_size = 1

device = "cpu"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# print(device)
#
# lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
#                                reduction = "mean")
#
# model = LSTM_net(batch_size=batch_size, device=device).to(device)
# x = torch.load(f"C:/Users/Marc/Desktop/network/network_LSTM-510.pt")
# # x = torch.load(r"C:\Users\Marc\Desktop\network\networks_515\network_LSTM-loss-epk-94.pt")
# model.load_state_dict(x)
# model.to(device)
#
#
# val_tester(run = "1", network = "2", model = model, lossFunc = lossFunc, device = device)

device = "cpu"

model = Unet_leaky_lstm(n_channels=1, batch_size=batch_size, \
                        device=device).to(device)

x = torch.load(f"C:/Users/Marc/Desktop/network/network_SGD-acc-epk-21.pt")
model.load_state_dict(x)
model.to(device)

lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
                               reduction = "mean")



# thenos, linear_model

val_tester(run = "1", network = "2", model = model, lossFunc = lossFunc, device = device)
