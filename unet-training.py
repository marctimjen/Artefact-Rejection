import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
import numpy as np
import time

from LoaderPACK.Unet import Unet
from LoaderPACK.Loader import load_whole_data, load_shuffle_5_min


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Set up the dataloaders:


# the first loader is for loading the different time-series data
# load_file = load_whole_data(path = "C:/Users/Marc/Desktop/model_data",
#                             ind = [i for i in range(1, 310 + 1)])

load_file = load_whole_data(path = "C:/Users/Marc/Desktop/model_data",
                            ind = [i for i in range(1, 4)])


file_loader = torch.utils.data.DataLoader(load_file,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=0)



lossFunc = nn.MSELoss()

# https://pytorch.org/docs/stable/generated/torch.quantile.html


# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
# bineær loss function - hvor man kan vægte klasserne.



model = Unet(n_channels = 1, n_classes = 2).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
train_loss = []

batch_size = 10

i = 0

for file in file_loader:

    print(i)
    i += 1

    # the second loader is for loading the random timed 5-mins intervals
    load_series = load_shuffle_5_min(file, device)

    series_loader = torch.utils.data.DataLoader(load_series,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0)

    for series in series_loader:
        ind, tar, chan = series
        ind = ind.reshape(ind.shape[0], 1, 15000)
        tar = tar.reshape(ind.shape[0], 1, 15000)
        y_pred = model(ind)
        model.zero_grad()
        loss = lossFunc(y_pred, tar)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())



print(train_loss)


# Med RNN - først træne med random derefter kører sequentielt ved træning af RNN'en.
