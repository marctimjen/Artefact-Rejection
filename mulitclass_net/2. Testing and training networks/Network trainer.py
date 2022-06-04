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
sys.path.append("../..") # adds higher directory to python modules path

from LoaderPACK.Unet_leaky import Unet_leaky_lstm_elec
from LoaderPACK.Loader import shuffle_5min
from LoaderPACK.trainer import net_multi_train
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
    n_samples = 11141 - 1 # the defualt amount of samples minus 1

    train_load_file = shuffle_5min(path = train_path,
                                   series_dict = 'train_series_length.pickle',
                                   size = (195, 22, 2060000),
                                   device = device,
                                   length=n_samples)

    train_loader = torch.utils.data.DataLoader(train_load_file,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True)

    val_load_file = shuffle_5min(path = val_path,
                                 series_dict = 'val_series_length.pickle',
                                 size = (28, 22, 549200),
                                 device = device,
                                 seed = 42)

    val_loader = torch.utils.data.DataLoader(val_load_file,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             drop_last=True)

    nEpoch = 50
    base_lr = 1.2 # where we start the learning rate
    max_lr = 1.3 # where the learning rate is supposed to end
    weight_decay = 0
    step_size_up = (n_samples/batch_size)*5 + 1

    model = Unet_leaky_lstm_elec(n_channels=1, batch_size=batch_size, \
                            device=device).to(device)

    optimizer = SGD(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1.25, 6.67, 16.67, 1.]).to(device),
                                   reduction = "mean")

    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                         step_size_up=step_size_up,
                         cycle_momentum=False)

    smooth = 0.05

    params = {"optimizer":"SGD", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "optimizor_weight_decay":weight_decay,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1.25, 6.67, 16.67, 1.],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky_lstm_elec", "scheduler":"CyclicLR",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "scheduler_cycle_momentum":False,
              "scheduler_step_size_up":step_size_up,
              "smooting_loss":smooth}

    run[f"{net_name}/parameters"] = params

    net_multi_train(device = device,
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
                    scheduler = scheduler)


def net_ADAM(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    net_name = "network_ADAM"

    batch_size = 10
    n_samples = 11141 - 1 # the defualt amount of samples minus 1

    train_load_file = shuffle_5min(path = train_path,
                                   series_dict = 'train_series_length.pickle',
                                   size = (195, 22, 2060000),
                                   device = device,
                                   length=n_samples)

    train_loader = torch.utils.data.DataLoader(train_load_file,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=True)

    val_load_file = shuffle_5min(path = val_path,
                                 series_dict = 'val_series_length.pickle',
                                 size = (28, 22, 549200),
                                 device = device,
                                 seed = 42)

    val_loader = torch.utils.data.DataLoader(val_load_file,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             drop_last=True)

    nEpoch = 50
    base_lr = 0.007 # where we start the learning rate
    max_lr = 0.013 # where the learning rate is supposed to end
    weight_decay = 0
    step_size_up = (n_samples/batch_size)*5 + 1

    model = Unet_leaky_lstm_elec(n_channels=1, batch_size=batch_size, \
                            device=device).to(device)

    optimizer = Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1.25, 6.67, 16.67, 1.]).to(device),
                                   reduction = "mean")

    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                         step_size_up=step_size_up,
                         cycle_momentum=False)

    smooth = 0.05

    params = {"optimizer":"SGD", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "optimizor_weight_decay":weight_decay,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1.25, 6.67, 16.67, 1.],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky_lstm_elec", "scheduler":"CyclicLR",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "scheduler_cycle_momentum":False,
              "scheduler_step_size_up":step_size_up,
              "smooting_loss":smooth}

    run[f"{net_name}/parameters"] = params

    net_multi_train(device = device,
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

    if device == "cpu":
        fl = torch.FloatTensor
        it = torch.LongTensor
    else:
        fl = torch.cuda.FloatTensor
        it = torch.cuda.LongTensor

    # core = torch.cuda.device_count()
    core = 1

    networks = [net_SGD] # net_ADAM

    cuda_dict = dict()
    cuda_dict[0] = networks

    # for i in range(core):
    #     cuda_dict[i] = []
    #
    # for i in range(len(networks)):
    #     cuda_dict[i % core].append(networks[i]) # i % core

        #"/home/tyson/model_data/train_model_data"
        # "C:/Users/Marc/Desktop/model_data/train_model_data"

    train_path = "/home/tyson/data/train_model_data"
    val_path = "/home/tyson/data/val_model_data"
    # train_path = r"C:\Users\Marc\Desktop\data\train_model_data"
    # val_path = r"C:\Users\Marc\Desktop\data\val_model_data"

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
