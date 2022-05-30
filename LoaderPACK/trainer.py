import neptune.new as neptune
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import re

import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Accuarcy_finder import Accuarcy_find
from LoaderPACK.Accuarcy_upload import Accuarcy_upload


def net_train(device,
              fl,
              it,
              net_name,
              model,
              optimizer,
              lossFunc,
              nEpoch,
              smooth,
              train_loader,
              val_loader,
              run,
              path,
              scheduler = None):
    """
    This function is used for training the hyper-optimized networks.

    Args:
        device (torch device): whihc device to train the network on.
        fl (tensor type): which type of tensor is used
                                                (differes depending on device)
        it (tensor type): which type of tensor is used
                                                (differes depending on device)
        net_name (str): the name of the network
        model (torch model): the network model used
        optimizer (torch optimizor): the optimizor used for the training
        lossFunc (torch loss funciton): the loss function used
        nEpoch (int): amount of epochs used for training
        smooth (float): amount of smoothing used for the recorded loss
        train_loader (torch loader): data loader for training
        val_loader (torch loader): data loader for validation
        run (neptune run instance): to log the information duing training and
                                    validation
        path (str): path for saving networks
        scheduler = None (torch scheduler): if given the scheduler will be used
                                            to update the learning rate

    Produced files:
        This function creates a folder at path, using the number of the neptune
        run instance. In this folder networks will be saved when new better
        accuracy or loss is achived. Also the final network will be saved in
        this folder.
    """

    valid_loss, train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    first_loss_save = True
    # save the loss of the network before training

    try: # test if the optimizor contain momentum
        moment = optimizer.param_groups[0]['momentum']
        moment = True
    except:
        moment = False

    # make dir to save the networks in
    str_run = run.get_run_url()
    m = re.match(r".+-(\d+)", str_run) # this correlates the name of the network
                                  # with the neptune ai name.
    run_nr = m.group(1)

    new_path = os.path.join(path, f"networks_{run_nr}")
    os.mkdir(new_path)

    path = path + f"networks_{run_nr}/"



    lowest_val_loss = float("inf") # the best loss obtained during training
    best_acc = -float("inf") # the best accuarcy during training

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        run[f"{net_name}/learning_rate"].log(optimizer.param_groups[0]['lr'])

        if moment:
            run[f"{net_name}/momentum"].log(
                                          optimizer.param_groups[0]['momentum'])

        t_mat = torch.zeros(2, 2) # save confusion matrix
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)

        for series in train_loader:
            ind, tar, chan = series
            model.zero_grad()  # clear the gradients before each instance
            y_pred = model(ind)
            pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(pred, target)

            if first_loss_save: # save loss before training
                run[f"{net_name}/train_loss_pr_file"].log(loss)
                run[f"{net_name}/smooth_train_loss_pr_file"].log(loss)
                t_sm_loss = loss.item()

                run[f"{net_name}/validation_loss_pr_file"].log(loss)
                run[f"{net_name}/smooth_val_loss_pr_file"].log(loss)
                v_sm_loss = loss.item()
                first_loss_save = False

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
            train_acc = torch.cat((train_acc, acc.view(1)))
            t_mat = t_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

            if scheduler: # update the value of the scheduler
                scheduler.step()

        # log the mean training loss
        run[f"{net_name}/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))

        sm_loss = np.mean(np.array(train_loss)) * smooth \
                            + (1-smooth) * t_sm_loss

        t_sm_loss = sm_loss
        # log the smoothed loss
        run[f"{net_name}/smooth_train_loss_pr_file"].log(sm_loss)

        train_loss = []

        run[f"{net_name}/train_acc_pr_file"].log(torch.mean(train_acc))
        train_acc = torch.tensor([]).to(device)

        run[f"{net_name}/matrix/train_confusion_matrix_pr_file"].log(t_mat)
        Accuarcy_upload(run, t_mat, total_pos, total_neg,
                        f"{net_name}", "train")

        v_mat = torch.zeros(2,2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)


        for series in val_loader:
            ind, tar, chan = series
            y_pred = model(ind)
            pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(pred, target)
            valid_loss.append(loss.item())

            acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
            valid_acc = torch.cat((valid_acc, acc.view(1)))
            v_mat = v_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

        avg_val_loss = np.mean(np.array(valid_loss))
        run[f"{net_name}/validation_loss_pr_file"].log(avg_val_loss)

        sm_loss = np.mean(np.array(valid_loss)) * smooth \
                            + (1-smooth) * v_sm_loss

        v_sm_loss = sm_loss

        run[f"{net_name}/smooth_val_loss_pr_file"].log(sm_loss)
        valid_loss = []

        avg_val_acc = torch.mean(valid_acc)
        run[f"{net_name}/val_acc_pr_file"].log(avg_val_acc)
        valid_acc = torch.tensor([]).to(device)

        if avg_val_loss < lowest_val_loss and best_acc < avg_val_acc \
            and iEpoch >= 5:
            # scenario when both the loss and accuarcy is better
            torch.save(model.state_dict(),
                       path + f"{net_name}-epk-{iEpoch}.pt")
            lowest_val_loss = avg_val_loss
            best_acc = avg_val_acc

        elif avg_val_loss < lowest_val_loss and iEpoch >= 5:
            # when only the loss is decreased
            torch.save(model.state_dict(),
                       path + f"{net_name}-loss-epk-{iEpoch}.pt")
            lowest_val_loss = avg_val_loss
        elif best_acc < avg_val_acc and iEpoch >= 5:
            # when only the accuracy is increased
            torch.save(model.state_dict(),
                       path + f"{net_name}-acc-epk-{iEpoch}.pt")
            best_acc = avg_val_acc

        run[f"{net_name}/matrix/val_confusion_matrix_pr_file"].log(v_mat)
        Accuarcy_upload(run, v_mat, total_pos, total_neg, f"{net_name}", "val")

    # save the final network
    torch.save(model.state_dict(), path + f"final-{net_name}-{run_nr}.pt")
    run.stop()


def net_train_combo(device,
                    fl,
                    it,
                    net_name,
                    model,
                    optimizer,
                    lossFunc,
                    nEpoch,
                    smooth,
                    train_loader,
                    val_loader,
                    run,
                    path,
                    scheduler = None):
    """
    This function is used for training the hyper-optimized networks.

    Args:
        device (torch device): whihc device to train the network on.
        fl (tensor type): which type of tensor is used
                                                (differes depending on device)
        it (tensor type): which type of tensor is used
                                                (differes depending on device)
        net_name (str): the name of the network
        model (torch model): the network model used
        optimizer (torch optimizor): the optimizor used for the training
        lossFunc (torch loss funciton): the loss function used
        nEpoch (int): amount of epochs used for training
        smooth (float): amount of smoothing used for the recorded loss
        train_loader (torch loader): data loader for training
        val_loader (torch loader): data loader for validation
        run (neptune run instance): to log the information duing training and
                                    validation
        path (str): path for saving networks
        scheduler = None (torch scheduler): if given the scheduler will be used
                                            to update the learning rate

    Produced files:
        This function creates a folder at path, using the number of the neptune
        run instance. In this folder networks will be saved when new better
        accuracy or loss is achived. Also the final network will be saved in
        this folder.
    """

    valid_loss, train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    first_loss_save = True
    # save the loss of the network before training

    try: # test if the optimizor contain momentum
        moment = optimizer.param_groups[0]['momentum']
        moment = True
    except:
        moment = False

    # make dir to save the networks in
    str_run = run.get_run_url()
    m = re.match(r".+-(\d+)", str_run) # this correlates the name of the network
                                  # with the neptune ai name.
    run_nr = m.group(1)

    new_path = os.path.join(path, f"networks_{run_nr}")
    os.mkdir(new_path)

    path = path + f"networks_{run_nr}/"



    lowest_val_loss = float("inf") # the best loss obtained during training
    best_acc = -float("inf") # the best accuarcy during training

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        run[f"{net_name}/learning_rate"].log(optimizer.param_groups[0]['lr'])

        if moment:
            run[f"{net_name}/momentum"].log(
                                          optimizer.param_groups[0]['momentum'])

        t_mat = torch.zeros(2, 2) # save confusion matrix
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)

        for series in train_loader:
            ind, tar, chan = series
            model.zero_grad()  # clear the gradients before each instance
            y_pred = model(ind)
            pred = y_pred[:, 1].type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(pred, target)

            if first_loss_save: # save loss before training
                run[f"{net_name}/train_loss_pr_file"].log(loss)
                run[f"{net_name}/smooth_train_loss_pr_file"].log(loss)
                t_sm_loss = loss.item()

                run[f"{net_name}/validation_loss_pr_file"].log(loss)
                run[f"{net_name}/smooth_val_loss_pr_file"].log(loss)
                v_sm_loss = loss.item()
                first_loss_save = False

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
            train_acc = torch.cat((train_acc, acc.view(1)))
            t_mat = t_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

            if scheduler: # update the value of the scheduler
                scheduler.step()

        # log the mean training loss
        run[f"{net_name}/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))

        sm_loss = np.mean(np.array(train_loss)) * smooth \
                            + (1-smooth) * t_sm_loss

        t_sm_loss = sm_loss
        # log the smoothed loss
        run[f"{net_name}/smooth_train_loss_pr_file"].log(sm_loss)

        train_loss = []

        run[f"{net_name}/train_acc_pr_file"].log(torch.mean(train_acc))
        train_acc = torch.tensor([]).to(device)

        run[f"{net_name}/matrix/train_confusion_matrix_pr_file"].log(t_mat)
        Accuarcy_upload(run, t_mat, total_pos, total_neg,
                        f"{net_name}", "train")

        v_mat = torch.zeros(2,2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)


        for series in val_loader:
            ind, tar, chan = series
            y_pred = model(ind)
            pred = y_pred[:, 1].type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(pred, target)
            valid_loss.append(loss.item())

            acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
            valid_acc = torch.cat((valid_acc, acc.view(1)))
            v_mat = v_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

        avg_val_loss = np.mean(np.array(valid_loss))
        run[f"{net_name}/validation_loss_pr_file"].log(avg_val_loss)

        sm_loss = np.mean(np.array(valid_loss)) * smooth \
                            + (1-smooth) * v_sm_loss

        v_sm_loss = sm_loss

        run[f"{net_name}/smooth_val_loss_pr_file"].log(sm_loss)
        valid_loss = []

        avg_val_acc = torch.mean(valid_acc)
        run[f"{net_name}/val_acc_pr_file"].log(avg_val_acc)
        valid_acc = torch.tensor([]).to(device)

        if avg_val_loss < lowest_val_loss and best_acc < avg_val_acc \
            and iEpoch >= 5:
            # scenario when both the loss and accuarcy is better
            torch.save(model.state_dict(),
                       path + f"{net_name}-epk-{iEpoch}.pt")
            lowest_val_loss = avg_val_loss
            best_acc = avg_val_acc

        elif avg_val_loss < lowest_val_loss and iEpoch >= 5:
            # when only the loss is decreased
            torch.save(model.state_dict(),
                       path + f"{net_name}-loss-epk-{iEpoch}.pt")
            lowest_val_loss = avg_val_loss
        elif best_acc < avg_val_acc and iEpoch >= 5:
            # when only the accuracy is increased
            torch.save(model.state_dict(),
                       path + f"{net_name}-acc-epk-{iEpoch}.pt")
            best_acc = avg_val_acc

        run[f"{net_name}/matrix/val_confusion_matrix_pr_file"].log(v_mat)
        Accuarcy_upload(run, v_mat, total_pos, total_neg, f"{net_name}", "val")

    # save the final network
    torch.save(model.state_dict(), path + f"final-{net_name}-{run_nr}.pt")
    run.stop()
