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

from LoaderPACK.Unet_leaky import Unet_leaky
from LoaderPACK.Loader import load_whole_data, load_shuffle_5_min
from LoaderPACK.Accuarcy_finder import Accuarcy_find
from LoaderPACK.Accuarcy_upload import Accuarcy_upload
from multiprocessing import Process

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def net_SGD1(device, fl, it, train_file_loader, val_file_loader):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    valid_loss, train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    avg_train_loss, avg_valid_loss = [], []

    model = Unet_leaky(n_channels=1, n_classes=2).to(device)
    optimizer = SGD(model.parameters(), lr=0.001)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
                                   reduction = "mean")

    nEpoch = 100
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=9,
                         step_size_up=nEpoch-1, cycle_momentum=False)

    batch_size = 20

    params = {"optimizer":"SGD", "batch_size":batch_size,
              "optimizer_learning_rate": 0.001,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1, 5],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky", "scheduler":"CyclicLR",
              "scheduler_base_lr":0.001, "scheduler_max_lr":9,
              "scheduler_cycle_momentum":False,
              "scheduler_step_size_up":nEpoch-1}

    run[f"network_SGD/parameters"] = params


    first_train = True
    first_val = True

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        run[f"network_SGD/learning_rate"].log(optimizer.param_groups[0]['lr'])

        t_mat = torch.zeros(2, 2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)

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
                pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
                target = tar.view(-1).type(it)
                loss = lossFunc(pred, target)
                if first_train:
                    run[f"network_SGD/train_loss_pr_file"].log(loss)
                    first_train = False
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

                acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
                train_acc = torch.cat((train_acc, acc.view(1)))
                t_mat = t_mat + mat
                total_pos = total_pos + tot_p_g
                total_neg = total_neg + tot_n_g

        run[f"network_SGD/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))
        train_loss = []

        run[f"network_SGD/train_acc_pr_file"].log(torch.mean(train_acc))
        train_acc = torch.tensor([]).to(device)

        run[f"network_SGD/matrix/train_confusion_matrix_pr_file"].log(t_mat)
        Accuarcy_upload(run, t_mat, total_pos, total_neg, "network_SGD", "train")

        v_mat = torch.zeros(2,2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)

        for file in val_file_loader:
            load_series = load_shuffle_5_min(file, device)

            series_loader = torch.utils.data.DataLoader(load_series,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=0)

            for series in series_loader:
                ind, tar, chan = series
                y_pred = model(ind)
                pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
                target = tar.view(-1).type(it)
                loss = lossFunc(pred, target)
                if first_val:
                    run[f"network_SGD/validation_loss_pr_file"].log(loss)
                    first_val = False
                valid_loss.append(loss.item())

                acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
                valid_acc = torch.cat((valid_acc, acc.view(1)))
                v_mat = v_mat + mat
                total_pos = total_pos + tot_p_g
                total_neg = total_neg + tot_n_g

        run[f"network_SGD/validation_loss_pr_file"].log(
                                                  np.mean(np.array(valid_loss)))
        valid_loss = []

        run[f"network_SGD/val_acc_pr_file"].log(torch.mean(valid_acc))
        valid_acc = torch.tensor([]).to(device)

        run[f"network_SGD/matrix/val_confusion_matrix_pr_file"].log(v_mat)
        Accuarcy_upload(run, v_mat, total_pos, total_neg, "network_SGD", "val")

        scheduler.step()
    run.stop()




def net_ADAM1(device, fl, it, train_file_loader, val_file_loader):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

    valid_loss, train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    avg_train_loss, avg_valid_loss = [], []

    model = Unet_leaky(n_channels=1, n_classes=2).to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
                                   reduction = "mean")

    nEpoch = 100
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.5,
                         step_size_up=nEpoch-1, cycle_momentum=False)

    batch_size = 20

    params = {"optimizer":"Adam", "batch_size":batch_size,
              "optimizer_learning_rate": 0.0001,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1, 5],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky", "scheduler":"CyclicLR",
              "scheduler_cycle_momentum":False,
              "scheduler_base_lr":0.0001, "scheduler_max_lr":0.5,
              "scheduler_step_size_up":nEpoch-1}

    run[f"network_ADAM/parameters"] = params

    first_train = True
    first_val = True

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        run[f"network_ADAM/learning_rate"].log(optimizer.param_groups[0]['lr'])

        t_mat = torch.zeros(2, 2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)

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
                pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
                target = tar.view(-1).type(it)
                loss = lossFunc(pred, target)
                loss.backward()
                if first_train:
                    run[f"network_ADAM/train_loss_pr_file"].log(loss)
                    first_train = False
                optimizer.step()
                train_loss.append(loss.item())

                acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
                train_acc = torch.cat((train_acc, acc.view(1)))
                t_mat = t_mat + mat
                total_pos = total_pos + tot_p_g
                total_neg = total_neg + tot_n_g
                #print(tot_n)

            #print(total_neg_train)
        run[f"network_ADAM/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))
        train_loss = []

        run[f"network_ADAM/train_acc_pr_file"].log(torch.mean(train_acc))
        train_acc = torch.tensor([]).to(device)

        run[f"network_ADAM/matrix/train_confusion_matrix_pr_file"].log(t_mat)
        Accuarcy_upload(run, t_mat, total_pos, total_neg, "network_ADAM", "train")

        v_mat = torch.zeros(2,2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)

        for file in val_file_loader:
            load_series = load_shuffle_5_min(file, device)

            series_loader = torch.utils.data.DataLoader(load_series,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=0)

            for series in series_loader:
                ind, tar, chan = series
                y_pred = model(ind)
                pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
                target = tar.view(-1).type(it)
                loss = lossFunc(pred, target)
                if first_val:
                    run[f"network_ADAM/validation_loss_pr_file"].log(loss)
                    first_val = False
                valid_loss.append(loss.item())

                acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
                valid_acc = torch.cat((valid_acc, acc.view(1)))
                v_mat = v_mat + mat
                total_pos = total_pos + tot_p_g
                total_neg = total_neg + tot_n_g

        run[f"network_ADAM/validation_loss_pr_file"].log(
                                                  np.mean(np.array(valid_loss)))
        valid_loss = []

        run[f"network_ADAM/val_acc_pr_file"].log(torch.mean(valid_acc))
        valid_acc = torch.tensor([]).to(device)

        run[f"network_ADAM/matrix/val_confusion_matrix_pr_file"].log(v_mat)
        Accuarcy_upload(run, v_mat, total_pos, total_neg, "network_ADAM", "val")

        scheduler.step()

    run.stop()



def net_starter(nets, device, fl, it, train_file_loader, val_file_loader):
    for net in nets:
        pr1 = mp.Process(target=net, args = (device, fl, it,
                                                train_file_loader,
                                                val_file_loader,))
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


    # Set up the datasets
    np.random.seed(42)



    train_set = random.sample(range(1, 195 + 1), 100)

    train_load_file = load_whole_data(path = "/home/tyson/model_data/train_model_data",
                                      ind = train_set,
                                      series_dict = 'train_series_length.pickle')

    # train_load_file = load_whole_data(path = "C:/Users/Marc/Desktop/model_data",
    #                                   ind = train_set)


    train_file_loader = torch.utils.data.DataLoader(train_load_file,
                                                    batch_size=1,
                                                    shuffle=True,
                                                    num_workers=0)


    val_set = random.sample(range(1, 28 + 1), 20)

    val_load_file = load_whole_data(path = "/home/tyson/model_data/val_model_data",
                                    ind = val_set,
                                    series_dict = 'val_series_length.pickle')

    # val_load_file = load_whole_data(path = "C:/Users/Marc/Desktop/model_data",
    #                                 ind = val_set)

    val_file_loader = torch.utils.data.DataLoader(val_load_file,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  num_workers=0)

    #token = os.getenv('Neptune_api')
    #run = neptune.init(
    #    project="NTLAB/artifact-rej-scalp",
    #    api_token=token,
    #)

    core = torch.cuda.device_count()

    networks = [net_SGD1, net_ADAM1]

    cuda_dict = dict()
    for i in range(core):
        cuda_dict[i] = []

    for i in range(len(networks)):
        cuda_dict[i % core].append(networks[i])

    pres = []
    for i in range(core):
        pres.append(mp.Process(target=net_starter, args = (cuda_dict.get(i),
                                                           f"cuda:{i}",
                                                           fl, it,
                                                           train_file_loader,
                                                           val_file_loader,)))

    for process in pres:
        process.start()

    for process in pres:
        process.join()
