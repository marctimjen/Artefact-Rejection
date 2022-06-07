import neptune.new as neptune
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import numpy as np
import random
import math

import sys
sys.path.append("../..") # adds higher directory to python modules path

from LoaderPACK.LSTM_net import LSTM_net
from LoaderPACK.Loader import shuffle_5min
from LoaderPACK.trainer import net_train_combo
from LoaderPACK.Accuarcy_finder import Accuarcy_find
from LoaderPACK.Accuarcy_upload import Accuarcy_upload
from multiprocessing import Process

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

#PyTorch
BETA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 # weighted contribution of modified CE loss compared to Dice loss

# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
# https://www.sciencedirect.com/science/article/pii/S0895611121001750#eqn0040

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, ind, tar, cd_ratio = CE_RATIO, smooth=1, beta=BETA, eps=1e-9):

        #flatten label and prediction tensors
        inputs = ind.reshape(-1)
        targets = tar.reshape(-1)

        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        out = - (beta * ((targets * torch.log(inputs)) + ((1 - beta) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (cd_ratio * weighted_ce) - ((1 - cd_ratio) * dice)

        return combo


def net_LSTM(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    net_name = "network_LSTM"

    batch_size = 50
    n_samples =  3500 # the defualt amount of samples minus 1

    train_load_file = shuffle_5min(path = train_path,
                                   series_dict = 'train_series_length.pickle',
                                   size = (195, 22, 2060000),
                                   device = device,
                                   length = n_samples)


    train_loader = torch.utils.data.DataLoader(train_load_file,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True)

    val_load_file = shuffle_5min(path = val_path,
                                 series_dict = 'val_series_length.pickle',
                                 size = (28, 22, 549200),
                                 device = device,
                                 seed = 42,
                                 length = 500)


    val_loader = torch.utils.data.DataLoader(val_load_file,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             drop_last=True)


    nEpoch = 100
    base_lr = 0.075 # The base lr

    model = LSTM_net(batch_size=batch_size, device=device).to(device)

    optimizer = Adam(model.parameters(), lr=base_lr)
    #lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
    #                               reduction = "mean")

    lossFunc = ComboLoss()

    smooth = 0.05

    params = {"optimizer":"Adam", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "loss_function":"combo",
              "model":"LSTM_net", "smooting_loss":smooth}

    run[f"{net_name}/parameters"] = params

    net_train_combo(device = device,
                    fl = fl, it = it,
                    net_name = net_name,
                    model = model,
                    optimizer = optimizer,
                    lossFunc = lossFunc,
                    nEpoch = nEpoch,
                    smooth = smooth,
                    train_loader = train_loader,
                    val_loader = val_loader,
                    run = run,
                    path = "/home/tyson/network/", #"C:/Users/Marc/Desktop/network/",
                    clip = True,
                    scheduler = None)


def net_LSTM_lr(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    net_name = "network_LSTM"

    batch_size = 30
    n_samples = 30 # the defualt amount of samples minus 1

    train_load_file = shuffle_5min(path = train_path,
                                   series_dict = 'train_series_length.pickle',
                                   size = (195, 22, 2060000),
                                   device = device,
                                   length = n_samples)


    train_loader = torch.utils.data.DataLoader(train_load_file,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True)

    val_load_file = shuffle_5min(path = val_path,
                                 series_dict = 'val_series_length.pickle',
                                 size = (28, 22, 549200),
                                 device = device,
                                 seed = 42,
                                 length = 30)


    val_loader = torch.utils.data.DataLoader(val_load_file,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             drop_last=True)


    nEpoch = 100
    base_lr = 0.025 # where we start the learning rate

    base_lr = 0.01 # where we start the learning rate
    max_lr = 2 # where the learning rate is supposed to end

    model = LSTM_net(batch_size=batch_size, device=device).to(device)

    optimizer = Adam(model.parameters(), lr=base_lr)
    #lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
    #                               reduction = "mean")

    lossFunc = ComboLoss()

    lam = lambda x: math.exp(x*math.log(max_lr / base_lr) \
                                / (nEpoch*n_samples / batch_size))

    scheduler = LambdaLR(optimizer, lam)

    smooth = 0.05

    params = {"optimizer":"Adam", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "loss_function":"combo",
              "model":"LSTM_net", "scheduler":"Exponential",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "smooting_loss":smooth}

    run[f"{net_name}/parameters"] = params

    net_train_combo(device = device,
                    fl = fl, it = it,
                    net_name = net_name,
                    model = model,
                    optimizer = optimizer,
                    lossFunc = lossFunc,
                    nEpoch = nEpoch,
                    smooth = smooth,
                    train_loader = train_loader,
                    val_loader = val_loader,
                    run = run,
                    path = "C:/Users/Marc/Desktop/network/", #"/home/tyson/network/",
                    scheduler = scheduler)




def net_starter(nets, device, fl, it, train_path, val_path):
    for net in nets:
        pr1 = mp.Process(target=net, args = (device, fl, it,
                                                train_path,
                                                val_path,))
        pr1.start()
        pr1.join()

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if str(device) == "cpu":
        fl = torch.FloatTensor
        it = torch.LongTensor
    else:
        fl = torch.cuda.FloatTensor
        it = torch.cuda.LongTensor

    # core = torch.cuda.device_count()
    core = 2
    networks = [net_LSTM]

    cuda_dict = dict()
    # cuda_dict[core] = networks

    cuda_dict[1] = networks

    # for i in range(core):
    #     cuda_dict[i] = []
    #
    # for i in range(len(networks)):
    #     cuda_dict[i % core].append(networks[i]) # i % core

        #"/home/tyson/model_data/train_model_data"
        # "C:/Users/Marc/Desktop/model_data/train_model_data"

    train_path = "/home/tyson/data_new/train_model_data"
    val_path = "/home/tyson/data_new/val_model_data"
    # train_path = r"C:\Users\Marc\Desktop\data_new\train_model_data"
    # val_path = r"C:\Users\Marc\Desktop\data_new\val_model_data"

    pres = []
    for i in range(1, core):
        pres.append(mp.Process(target=net_starter, args = (cuda_dict.get(i),
                                                           f"cuda:{i}",
                                                           fl, it,
                                                           train_path,
                                                           val_path,)))

    for process in pres:
        process.start()

    for process in pres:
        process.join()
