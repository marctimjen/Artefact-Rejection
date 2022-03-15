import neptune.new as neptune
import os
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

if device == "cpu":
    fl = torch.FloatTensor
    it = torch.LongTensor
else:
    fl = torch.cuda.FloatTensor
    it = torch.cuda.LongTensor


# Set up the dataloaders:


# the first loader is for loading the different time-series data
# load_file = load_whole_data(path = "C:/Users/Marc/Desktop/model_data",
#                             ind = [i for i in range(1, 310 + 1)])


# Set up the datasets
np.random.seed(42)


val_set, train_set = torch.utils.data.random_split(
                            np.random.randint(low = 1, high = 284, size = 200),
                            [20, 180],
                            generator=torch.Generator().manual_seed(42))


#train_load_file = load_whole_data(path = "/home/tyson/model_data",
#                                  ind = train_set)

train_load_file = load_whole_data(path = "C:/Users/Marc/Desktop/model_data",
                                  ind = train_set)


train_file_loader = torch.utils.data.DataLoader(train_load_file,
                                                batch_size=1,
                                                shuffle=True,
                                                num_workers=0)


#val_load_file = load_whole_data(path = "/home/tyson/model_data",
#                                ind = val_set)

val_load_file = load_whole_data(path = "C:/Users/Marc/Desktop/model_data",
                                ind = val_set)

val_file_loader = torch.utils.data.DataLoader(val_load_file,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=0)



# "C:/Users/Marc/Desktop/model_data"

#lossFunc = nn.MSELoss()
#lossFunc = nn.BCELoss()




# https://pytorch.org/docs/stable/generated/torch.quantile.html


# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
# bineær loss function - hvor man kan vægte klasserne.


# Strukturer træningen, så man får en figur man kan vise.
# Tænk over hvordan man træner: Forøg data/gør modellen større/mindre.
# Gør ting struktureret for at vise, at man kan anvende de metoder man har gjort.
# Start småt og byg større.
# False negative og False positives.
# Kig på data - hvor laver modellen fejl?

# https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html


# Cross entropy:
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss



# mobaXterm - til ssh
token = os.getenv('Neptune_api')
run = neptune.init(
    project="NTLAB/artifact-rej-scalp",
    api_token=token,
)


params = {"optimizer":"SGD", "optimizer_momentum": 0.9,
          "optimizer_learning_rate": 0.1, "loss_function":"CrossEntropyLoss",
          "loss_function_weights":[1, 10], "loss_function_reduction":"mean",
          "model":"Unet"}

run[f"network/parameters"] = params


valid_loss, train_loss = [], []
valid_acc, train_acc = [], []

avg_train_loss, avg_valid_loss = [], []

model = Unet(n_channels = 1, n_classes = 2).to(device)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 10.]).to(device), reduction = "mean")

batch_size = 10

nEpoch = 40



for iEpoch in range(nEpoch):
    print(f"Training epoch {iEpoch}")
    for file in train_file_loader:

        # the second loader is for loading the random timed 5-mins intervals
        load_series = load_shuffle_5_min(file, device)

        series_loader = torch.utils.data.DataLoader(load_series,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

        for series in series_loader:
            ind, tar, chan = series
            y_pred = model(ind)
            model.zero_grad()
            y_pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(y_pred, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

    avg_train_loss.append(w := (np.mean(np.array(train_loss))))
    run[f"network/train_loss_pr_file"].log(w)
    train_loss = []


    for file in val_file_loader:
        load_series = load_shuffle_5_min(file, device)

        series_loader = torch.utils.data.DataLoader(load_series,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)

        for series in series_loader:
            ind, tar, chan = series
            y_pred = model(ind)
            loss = lossFunc(y_pred, tar)
            valid_loss.append(loss.item())

    avg_valid_loss.append(w := (np.mean(np.array(valid_loss))))
    run[f"network/validation_loss_pr_file"].log(w)
    valid_loss = []




#import matplotlib.pyplot as plt

#plt.plot(avg_train_loss)
#plt.show()
#
#plt.plot(avg_valid_loss)
#plt.show()


# Med RNN - først træne med random derefter kører sequentielt ved træning af RNN'en.
run.stop()
