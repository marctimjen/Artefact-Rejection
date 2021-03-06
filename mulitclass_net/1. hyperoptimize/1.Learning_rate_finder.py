import neptune.new as neptune
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
import torch.multiprocessing as mp
import numpy as np
import random
import math

import sys
sys.path.append("../..") # adds higher directory to python modules path

from LoaderPACK.Unet_leaky import Unet_leaky, Unet_leaky_lstm_elec
from LoaderPACK.Loader import shuffle_5min
from LoaderPACK.Accuarcy_finder import Accuarcy_find_elec
from LoaderPACK.Accuarcy_upload import Accuarcy_upload_elec
from multiprocessing import Process

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def net_SGD1(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

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


    valid_loss, train_loss = [], []
    smooth_valid_loss, smooth_train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    nEpoch = 5
    base_lr = 0.2 # where we start the learning rate
    max_lr = 5 # where the learning rate is supposed to end

    model = Unet_leaky_lstm_elec(n_channels=1, batch_size=batch_size, \
                                 device=device).to(device)
    # model = Unet_leaky(n_channels=1, n_classes=2).to(device)
    optimizer = SGD(model.parameters(), lr=base_lr)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1.25, 6.67, 16.67, 1.]).to(device),
                                   reduction = "mean")

    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                         step_size_up=nEpoch*(n_samples/batch_size)-1,
                         cycle_momentum=False)
    # step_size_up is set so the learning rate is updated linearly

    smooth = 0.05

    params = {"optimizer":"SGD", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1.25, 6.67, 16.67, 1.],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky_lstm_elec", "scheduler":"CyclicLR",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "scheduler_cycle_momentum":False,
              "scheduler_step_size_up":nEpoch*(n_samples/batch_size)-1,
              "smooting_loss":smooth}

    run[f"network_SGD/parameters"] = params


    first_loss_save = True

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        for series in train_loader:
            run[f"network_SGD/learning_rate"].log(
                                                optimizer.param_groups[0]['lr'])

            t_mat = torch.zeros(3, 2, 2)
            total_pos, total_neg = torch.tensor(0), torch.tensor(0)

            ind, tar, chan = series
            optimizer.zero_grad()

            y_pred = model(ind)

            # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(y_pred, target)

            if first_loss_save:
                run[f"network_SGD/train_loss_pr_file"].log(loss)
                run[f"network_SGD/smooth_train_loss_pr_file"].log(loss)
                smooth_train_loss.append(loss.item())

                run[f"network_SGD/validation_loss_pr_file"].log(loss)
                run[f"network_SGD/smooth_val_loss_pr_file"].log(loss)
                smooth_valid_loss.append(loss.item())
                first_loss_save = False

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())


            acc, mat, tot_p_g, tot_n_g = Accuarcy_find_elec(y_pred, tar)
            train_acc = torch.cat((train_acc, acc.view(1)))
            t_mat = t_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

            run[f"network_SGD/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))

            train_loss = []

            sm_loss = loss.item() * smooth + (1-smooth) * smooth_train_loss[-1]
            smooth_train_loss.append(sm_loss)
            run[f"network_SGD/smooth_train_loss_pr_file"].log(sm_loss)

            run[f"network_SGD/train_acc_pr_file"].log(torch.mean(train_acc))
            train_acc = torch.tensor([]).to(device)

            run[f"network_SGD/matrix/train_confusion_matrix_pr_file"].log(t_mat)
            Accuarcy_upload_elec(run, t_mat, total_pos, total_neg,
                            "network_SGD", "train")

            v_mat = torch.zeros(3, 2, 2)
            total_pos, total_neg = torch.tensor(0), torch.tensor(0)


            for series in val_loader:
                ind, tar, chan = series
                with torch.no_grad():
                    y_pred = model(ind)
                # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
                target = tar.view(-1).type(it)
                loss = lossFunc(y_pred, target)
                valid_loss.append(loss.item())

                acc, mat, tot_p_g, tot_n_g = Accuarcy_find_elec(y_pred, tar)
                valid_acc = torch.cat((valid_acc, acc.view(1)))
                v_mat = v_mat + mat
                total_pos = total_pos + tot_p_g
                total_neg = total_neg + tot_n_g

            run[f"network_SGD/validation_loss_pr_file"].log(
                                                  np.mean(np.array(valid_loss)))

            sm_loss = np.mean(np.array(valid_loss)) * smooth \
                        + (1-smooth) * smooth_valid_loss[-1]

            smooth_valid_loss.append(sm_loss)
            run[f"network_SGD/smooth_val_loss_pr_file"].log(sm_loss)

            valid_loss = []

            run[f"network_SGD/val_acc_pr_file"].log(torch.mean(valid_acc))
            valid_acc = torch.tensor([]).to(device)

            run[f"network_SGD/matrix/val_confusion_matrix_pr_file"].log(v_mat)
            Accuarcy_upload_elec(run, v_mat, total_pos, total_neg,
                            "network_SGD", "val")
            scheduler.step()
    run.stop()

def net_SGD2(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

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


    valid_loss, train_loss = [], []
    smooth_valid_loss, smooth_train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    nEpoch = 5
    base_lr = 0.1 # where we start the learning rate
    max_lr = 9 # where the learning rate is supposed to end

    model = Unet_leaky_lstm_elec(n_channels=1, batch_size=batch_size, \
                                 device=device).to(device)
    # model = Unet_leaky(n_channels=1, n_classes=2).to(device)
    optimizer = SGD(model.parameters(), lr=base_lr)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1.25, 6.67, 16.67, 1.]).to(device),
                                   reduction = "mean")

    lam = lambda x: math.exp(x*math.log(max_lr / base_lr) \
                                / (nEpoch*n_samples / batch_size))
    scheduler = LambdaLR(optimizer, lam)
    # implement the exponential learning rate

    smooth = 0.05

    params = {"optimizer":"SGD", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1.25, 6.67, 16.67, 1.],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky_lstm_elec", "scheduler":"Exponential",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "smooting_loss":smooth}

    run[f"network_SGD/parameters"] = params


    first_loss_save = True

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        for series in train_loader:
            run[f"network_SGD/learning_rate"].log(
                                                optimizer.param_groups[0]['lr'])

            t_mat = torch.zeros(3, 2, 2)
            total_pos, total_neg = torch.tensor(0), torch.tensor(0)

            ind, tar, chan = series
            optimizer.zero_grad()

            y_pred = model(ind)

            # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(y_pred, target)

            if first_loss_save:
                run[f"network_SGD/train_loss_pr_file"].log(loss)
                run[f"network_SGD/smooth_train_loss_pr_file"].log(loss)
                smooth_train_loss.append(loss.item())

                run[f"network_SGD/validation_loss_pr_file"].log(loss)
                run[f"network_SGD/smooth_val_loss_pr_file"].log(loss)
                smooth_valid_loss.append(loss.item())
                first_loss_save = False

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())


            acc, mat, tot_p_g, tot_n_g = Accuarcy_find_elec(y_pred, tar)
            train_acc = torch.cat((train_acc, acc.view(1)))
            t_mat = t_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

            run[f"network_SGD/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))

            train_loss = []

            sm_loss = loss.item() * smooth + (1-smooth) * smooth_train_loss[-1]
            smooth_train_loss.append(sm_loss)
            run[f"network_SGD/smooth_train_loss_pr_file"].log(sm_loss)

            run[f"network_SGD/train_acc_pr_file"].log(torch.mean(train_acc))
            train_acc = torch.tensor([]).to(device)

            run[f"network_SGD/matrix/train_confusion_matrix_pr_file"].log(t_mat)
            Accuarcy_upload_elec(run, t_mat, total_pos, total_neg,
                            "network_SGD", "train")

            v_mat = torch.zeros(3, 2, 2)
            total_pos, total_neg = torch.tensor(0), torch.tensor(0)


            for series in val_loader:
                ind, tar, chan = series
                with torch.no_grad():
                    y_pred = model(ind)
                # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
                target = tar.view(-1).type(it)
                loss = lossFunc(y_pred, target)
                valid_loss.append(loss.item())

                acc, mat, tot_p_g, tot_n_g = Accuarcy_find_elec(y_pred, tar)
                valid_acc = torch.cat((valid_acc, acc.view(1)))
                v_mat = v_mat + mat
                total_pos = total_pos + tot_p_g
                total_neg = total_neg + tot_n_g

            run[f"network_SGD/validation_loss_pr_file"].log(
                                                  np.mean(np.array(valid_loss)))

            sm_loss = np.mean(np.array(valid_loss)) * smooth \
                        + (1-smooth) * smooth_valid_loss[-1]

            smooth_valid_loss.append(sm_loss)
            run[f"network_SGD/smooth_val_loss_pr_file"].log(sm_loss)

            valid_loss = []

            run[f"network_SGD/val_acc_pr_file"].log(torch.mean(valid_acc))
            valid_acc = torch.tensor([]).to(device)

            run[f"network_SGD/matrix/val_confusion_matrix_pr_file"].log(v_mat)
            Accuarcy_upload_elec(run, v_mat, total_pos, total_neg,
                            "network_SGD", "val")
            scheduler.step()
    run.stop()


def net_ADAM1(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

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


    valid_loss, train_loss = [], []
    smooth_valid_loss, smooth_train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    nEpoch = 5
    base_lr = 0.0000000001 # where we start the learning rate
    max_lr = 0.01 # where the learning rate is supposed to end

    model = Unet_leaky_lstm_elec(n_channels=1, batch_size=batch_size, \
                                 device=device).to(device)
    # model = Unet_leaky(n_channels=1, n_classes=2).to(device)
    optimizer = Adam(model.parameters(), lr=base_lr)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1.25, 6.67, 16.67, 1.]).to(device),
                                   reduction = "mean")

    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                         step_size_up=nEpoch*(n_samples/batch_size)-1,
                         cycle_momentum=False)
    # step_size_up is set so the learning rate is updated linearly

    smooth = 0.05

    params = {"optimizer":"Adam", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1.25, 6.67, 16.67, 1.],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky_lstm_elec", "scheduler":"CyclicLR",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "scheduler_cycle_momentum":False,
              "scheduler_step_size_up":nEpoch*(n_samples/batch_size)-1,
              "smooting_loss":smooth}

    run[f"network_ADAM/parameters"] = params


    first_loss_save = True

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        for series in train_loader:
            run[f"network_ADAM/learning_rate"].log(
                                                optimizer.param_groups[0]['lr'])

            t_mat = torch.zeros(3, 2, 2)
            total_pos, total_neg = torch.tensor(0), torch.tensor(0)

            ind, tar, chan = series
            optimizer.zero_grad()

            y_pred = model(ind)

            # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(y_pred, target)

            if first_loss_save:
                run[f"network_ADAM/train_loss_pr_file"].log(loss)
                run[f"network_ADAM/smooth_train_loss_pr_file"].log(loss)
                smooth_train_loss.append(loss.item())

                run[f"network_ADAM/validation_loss_pr_file"].log(loss)
                run[f"network_ADAM/smooth_val_loss_pr_file"].log(loss)
                smooth_valid_loss.append(loss.item())
                first_loss_save = False

            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()
            train_loss.append(loss.item())


            acc, mat, tot_p_g, tot_n_g = Accuarcy_find_elec(y_pred, tar)
            train_acc = torch.cat((train_acc, acc.view(1)))
            t_mat = t_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

            run[f"network_ADAM/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))

            train_loss = []

            sm_loss = loss.item() * smooth + (1-smooth) * smooth_train_loss[-1]
            smooth_train_loss.append(sm_loss)
            run[f"network_ADAM/smooth_train_loss_pr_file"].log(sm_loss)

            run[f"network_ADAM/train_acc_pr_file"].log(torch.mean(train_acc))
            train_acc = torch.tensor([]).to(device)

            run[f"network_ADAM/matrix/train_confusion_matrix_pr_file"].log(t_mat)
            Accuarcy_upload_elec(run, t_mat, total_pos, total_neg,
                            "network_ADAM", "train")

            v_mat = torch.zeros(3, 2, 2)
            total_pos, total_neg = torch.tensor(0), torch.tensor(0)


            for series in val_loader:
                ind, tar, chan = series
                with torch.no_grad():
                    y_pred = model(ind)
                # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
                target = tar.view(-1).type(it)
                loss = lossFunc(y_pred, target)
                valid_loss.append(loss.item())

                acc, mat, tot_p_g, tot_n_g = Accuarcy_find_elec(y_pred, tar)
                valid_acc = torch.cat((valid_acc, acc.view(1)))
                v_mat = v_mat + mat
                total_pos = total_pos + tot_p_g
                total_neg = total_neg + tot_n_g

            run[f"network_ADAM/validation_loss_pr_file"].log(
                                                  np.mean(np.array(valid_loss)))

            sm_loss = np.mean(np.array(valid_loss)) * smooth \
                        + (1-smooth) * smooth_valid_loss[-1]

            smooth_valid_loss.append(sm_loss)
            run[f"network_ADAM/smooth_val_loss_pr_file"].log(sm_loss)

            valid_loss = []

            run[f"network_ADAM/val_acc_pr_file"].log(torch.mean(valid_acc))
            valid_acc = torch.tensor([]).to(device)

            run[f"network_ADAM/matrix/val_confusion_matrix_pr_file"].log(v_mat)
            Accuarcy_upload_elec(run, v_mat, total_pos, total_neg,
                            "network_ADAM", "val")
            scheduler.step()
    run.stop()

def net_ADAM2(device, fl, it, train_path, val_path):

    token = os.getenv('Neptune_api')
    run = neptune.init(
        project="NTLAB/artifact-rej-scalp",
        api_token=token,
    )

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


    valid_loss, train_loss = [], []
    smooth_valid_loss, smooth_train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    nEpoch = 5
    base_lr = 0.0001 # where we start the learning rate
    max_lr = 0.7 # where the learning rate is supposed to end

    model = Unet_leaky_lstm_elec(n_channels=1, batch_size=batch_size, \
                                 device=device).to(device)
    # model = Unet_leaky(n_channels=1, n_classes=2).to(device)
    optimizer = Adam(model.parameters(), lr=base_lr)
    lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1.25, 6.67, 16.67, 1.]).to(device),
                                   reduction = "mean")

    lam = lambda x: math.exp(x*math.log(max_lr / base_lr) \
                                / (nEpoch*n_samples / batch_size))
    scheduler = LambdaLR(optimizer, lam)
    # implement the exponential learning rate

    smooth = 0.05

    params = {"optimizer":"Adam", "batch_size":batch_size,
              "optimizer_learning_rate": base_lr,
              "loss_function":"CrossEntropyLoss",
              "loss_function_weights":[1.25, 6.67, 16.67, 1.],
              "loss_function_reduction":"mean",
              "model":"Unet_leaky_lstm_elec", "scheduler":"Exponential",
              "scheduler_base_lr":base_lr, "scheduler_max_lr":max_lr,
              "smooting_loss":smooth}

    run[f"network_ADAM/parameters"] = params


    first_loss_save = True

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        for series in train_loader:
            run[f"network_ADAM/learning_rate"].log(
                                                optimizer.param_groups[0]['lr'])

            t_mat = torch.zeros(3, 2, 2)
            total_pos, total_neg = torch.tensor(0), torch.tensor(0)

            ind, tar, chan = series
            optimizer.zero_grad()

            y_pred = model(ind)

            # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(y_pred, target)

            if first_loss_save:
                run[f"network_ADAM/train_loss_pr_file"].log(loss)
                run[f"network_ADAM/smooth_train_loss_pr_file"].log(loss)
                smooth_train_loss.append(loss.item())

                run[f"network_ADAM/validation_loss_pr_file"].log(loss)
                run[f"network_ADAM/smooth_val_loss_pr_file"].log(loss)
                smooth_valid_loss.append(loss.item())
                first_loss_save = False

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())


            acc, mat, tot_p_g, tot_n_g = Accuarcy_find_elec(y_pred, tar)
            train_acc = torch.cat((train_acc, acc.view(1)))
            t_mat = t_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

            run[f"network_ADAM/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))

            train_loss = []

            sm_loss = loss.item() * smooth + (1-smooth) * smooth_train_loss[-1]
            smooth_train_loss.append(sm_loss)
            run[f"network_ADAM/smooth_train_loss_pr_file"].log(sm_loss)

            run[f"network_ADAM/train_acc_pr_file"].log(torch.mean(train_acc))
            train_acc = torch.tensor([]).to(device)

            run[f"network_ADAM/matrix/train_confusion_matrix_pr_file"].log(t_mat)
            Accuarcy_upload_elec(run, t_mat, total_pos, total_neg,
                            "network_ADAM", "train")

            v_mat = torch.zeros(3, 2, 2)
            total_pos, total_neg = torch.tensor(0), torch.tensor(0)


            for series in val_loader:
                ind, tar, chan = series
                with torch.no_grad():
                    y_pred = model(ind)
                # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
                target = tar.view(-1).type(it)
                loss = lossFunc(y_pred, target)
                valid_loss.append(loss.item())

                acc, mat, tot_p_g, tot_n_g = Accuarcy_find_elec(y_pred, tar)
                valid_acc = torch.cat((valid_acc, acc.view(1)))
                v_mat = v_mat + mat
                total_pos = total_pos + tot_p_g
                total_neg = total_neg + tot_n_g

            run[f"network_ADAM/validation_loss_pr_file"].log(
                                                  np.mean(np.array(valid_loss)))

            sm_loss = np.mean(np.array(valid_loss)) * smooth \
                        + (1-smooth) * smooth_valid_loss[-1]

            smooth_valid_loss.append(sm_loss)
            run[f"network_ADAM/smooth_val_loss_pr_file"].log(sm_loss)

            valid_loss = []

            run[f"network_ADAM/val_acc_pr_file"].log(torch.mean(valid_acc))
            valid_acc = torch.tensor([]).to(device)

            run[f"network_ADAM/matrix/val_confusion_matrix_pr_file"].log(v_mat)
            Accuarcy_upload_elec(run, v_mat, total_pos, total_neg,
                            "network_ADAM", "val")
            scheduler.step()
    run.stop()

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

    core = torch.cuda.device_count()

    #core = 1

    networks = [net_ADAM1] # net_SGD1, , net_SGD2, net_ADAM2

    cuda_dict = dict()
    # cuda_dict[core] = networks

    for i in range(core):
        cuda_dict[i] = []

    for i in range(len(networks)):
        cuda_dict[i % core].append(networks[i]) # i % core

        #"/home/tyson/model_data/train_model_data"
    # train_path = r"D:\Alt fra desk\data_1\train_model_data"
    # val_path = r"D:\Alt fra desk\data_1\val_model_data"

    train_path = "C:/Users/Marc/Desktop/model_data/train_model_data"
    val_path = "C:/Users/Marc/Desktop/model_data/val_model_data"

    # train_path = "/home/tyson/data_cutoff/train_model_data"
    # val_path = "/home/tyson/data_cutoff/val_model_data"

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
