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
from LoaderPACK.Loader import load_whole_data, load_shuffle_5_min
from LoaderPACK.Accuarcy_finder import Accuarcy_find
from LoaderPACK.Accuarcy_upload import Accuarcy_upload
from multiprocessing import Process
from LoaderPACK.trainer import net_train

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def net_SGD(device, train_file_loader, val_file_loader):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    model = Unet(n_channels=1, n_classes=2).to(device)
    optimizer = SGD(model.parameters(), lr=0.6)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
                                   reduction = "mean")

    nEpoch = 4
    batch_size = 10

    net_name = "SGD_net"

    params = {"optimizer":"SGD", "batch_size":batch_size,
              "optimizer_learning_rate": 0.6,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1, 5],
              "loss_function_reduction":"mean",
              "model":"Unet"}

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


def net_ADAM(device, train_file_loader, val_file_loader):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    model = Unet(n_channels=1, n_classes=2).to(device)
    optimizer = Adam(model.parameters(), lr=0.37)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
                                   reduction = "mean")

    nEpoch = 4
    batch_size = 10

    net_name = "ADAM_net"

    params = {"optimizer":"ADAM", "batch_size":batch_size,
              "optimizer_learning_rate": 0.37,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1, 5],
              "loss_function_reduction":"mean",
              "model":"Unet"}

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


def net_starter(nets, device, train_file_loader, val_file_loader):
    for net in nets:
        pr1 = mp.Process(target=net, args = (device,
                                             train_file_loader,
                                             val_file_loader,))
        pr1.start()
        pr1.join()


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Set up the datasets
    np.random.seed(42)



    #val_set, train_set = torch.utils.data.random_split(
    #                            random.sample(range(1, 226 + 1), 200), [26, 200],
    #                            generator=torch.Generator().manual_seed(42))

    val_set, train_set = torch.utils.data.random_split(
                                random.sample(range(1, 226 + 1), 5), [1, 4],
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


    core = torch.cuda.device_count()

    networks = [net_SGD, net_ADAM]

    cuda_dict = dict()
    for i in range(core):
        cuda_dict[i] = []

    for i in range(len(networks)):
        cuda_dict[i % core].append(networks[i])

    pres = []
    for i in range(core):
        pres.append(mp.Process(target=net_starter, args = (cuda_dict.get(i),
                                                           f"cuda:{i}",
                                                           train_file_loader,
                                                           val_file_loader,)))

    for process in pres:
        process.start()

    for process in pres:
        process.join()
