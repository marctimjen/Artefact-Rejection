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
import math

import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Unet_leaky import Unet_leaky, Unet_leaky_lstm
from LoaderPACK.Loader import shuffle_5min
from LoaderPACK.trainer import net_train
from LoaderPACK.Accuarcy_finder import Accuarcy_find
from LoaderPACK.Accuarcy_upload import Accuarcy_upload
from multiprocessing import Process

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def net_SGD(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    net_name = "network_SGD"

    batch_size = 10

    train_load_file = shuffle_5min(path = train_path,
                                   series_dict = 'train_series_length.pickle',
                                   size = (195, 22, 2060000),
                                   device = device)


    train_loader = torch.utils.data.DataLoader(train_load_file,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    val_load_file = shuffle_5min(path = val_path,
                                 series_dict = 'val_series_length.pickle',
                                 size = (28, 22, 549200),
                                 device = device,
                                 seed = 42)


    val_loader = torch.utils.data.DataLoader(val_load_file,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)


    nEpoch = 5
    base_lr = 0.216 # where we start the learning rate (min point)
    max_lr = 0.268 # where the learning rate is at the max point
    weight_decay = 0

    model = Unet_leaky_lstm(n_channels=1, batch_size=batch_size, \
                            device=device).to(device)

    optimizer = SGD(model.parameters(), lr=base_lr)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
                                   reduction = "mean")

    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                         step_size_up=nEpoch*(n_samples/batch_size)-1,
                         cycle_momentum=True, base_momentum=0.8,
                         max_momentum=0.9)

    smooth = 0.05

    params = {"optimizer":"SGD", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1, 5],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky_lstm", "scheduler":"CyclicLR",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "scheduler_cycle_momentum":True,
              "base_momentum":0.8, "max_momentum":0.9,
              "scheduler_step_size_up":nEpoch*(n_samples/batch_size)-1,
              "smooting_loss":smooth}

    run[f"{net_name}/parameters"] = params

    net_train(device = device,
              net_name = net_name,
              model = model,
              optimizer = optimizer,
              lossFunc = lossFunc,
              nEpoch = nEpoch,
              batch_size = batch_size,
              train_file_loader = train_file_loader,
              val_file_loader = val_file_loader,
              run = run,
              path = "C:/Users/Marc/Desktop/network/",
              scheduler = None)


def net_ADAM(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    net_name = "network_ADAM"

    batch_size = 10
    n_samples = 500 # how many samples do we collect

    train_load_file = shuffle_5min(path = train_path,
                                   series_dict = 'train_series_length.pickle',
                                   size = (195, 22, 2060000),
                                   device = device,
                                   length = n_samples)


    train_loader = torch.utils.data.DataLoader(train_load_file,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    val_load_file = shuffle_5min(path = val_path,
                                 series_dict = 'val_series_length.pickle',
                                 size = (28, 22, 549200),
                                 device = device,
                                 seed = 42,
                                 length = 50)


    val_loader = torch.utils.data.DataLoader(val_load_file,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0)


    nEpoch = 5
    base_lr = 0.0089 # where we start the learning rate (min point)
    max_lr = 0.013 # where the learning rate is at the max point
    weight_decay = 0.0001

    model = Unet_leaky_lstm(n_channels=1, batch_size=batch_size, \
                            device=device).to(device)
    # model = Unet_leaky(n_channels=1, n_classes=2).to(device)
    optimizer = Adam(model.parameters(), lr=0.004, weight_decay=weight_decay)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
                                   reduction = "mean")

    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                         step_size_up=(nEpoch*(n_samples/batch_size)/6),
                         cycle_momentum=False)
    # step_size_up is set so the learning rate is updated linearly

    smooth = 0.05

    params = {"optimizer":"ADAM", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "optimizer_weight_decay": weight_decay,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1, 5],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky_lstm", "scheduler":"CyclicLR",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "scheduler_cycle_momentum":False,
              "scheduler_step_size_up":(nEpoch*(n_samples/batch_size)/6),
              "smooting_loss":smooth}

    run[f"{net_name}/parameters"] = params


    net_train(device = device,
              fl = fl, it = it,
              net_name = net_name,
              model = model,
              optimizer = optimizer,
              lossFunc = lossFunc,
              nEpoch = nEpoch,
              batch_size = batch_size,
              train_loader = train_loader,
              val_loader = val_loader,
              run = run,
              path = "C:/Users/Marc/Desktop/network/",
              scheduler = None)



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

    if device == "cpu":
        fl = torch.FloatTensor
        it = torch.LongTensor
    else:
        fl = torch.cuda.FloatTensor
        it = torch.cuda.LongTensor

    core = torch.cuda.device_count()

    #core = 1

    #networks = [net_ADAM1] # net_SGD2, net_ADAM2, net_SGD1,

    networks = [net_SGD1, net_SGD2]

    cuda_dict = dict()
    # cuda_dict[core] = networks

    for i in range(core):
        cuda_dict[i] = []

    for i in range(len(networks)):
        cuda_dict[i % core].append(networks[i]) # i % core

        #"/home/tyson/model_data/train_model_data"
        # "C:/Users/Marc/Desktop/model_data/train_model_data"

    train_path = "/home/tyson/data_cutoff/train_model_data"
    val_path = "/home/tyson/data_cutoff/val_model_data"

    pres = []
    for i in range(core):
        pres.append(mp.Process(target=net_starter, args = (cuda_dict.get(i),
                                                           f"cuda:{i}",
                                                           fl, it,
                                                           train_path,
                                                           val_path,)))

    for process in pres:
        process.start()

    for process in pres:
        process.join()
