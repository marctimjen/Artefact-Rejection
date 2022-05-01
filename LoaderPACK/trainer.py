import neptune.new as neptune
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp
import numpy as np

import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Accuarcy_finder import Accuarcy_find
from LoaderPACK.Accuarcy_upload import Accuarcy_upload


def net_train(device,
              net_name,
              model,
              optimizer,
              lossFunc,
              nEpoch,
              train_loader,
              val_loader,
              run,
              path,
              scheduler = None):

    if device == "cpu":
        fl = torch.FloatTensor
        it = torch.LongTensor
    else:
        fl = torch.cuda.FloatTensor
        it = torch.cuda.LongTensor

    valid_loss, train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    avg_train_loss, avg_valid_loss = [], []

    first_train = True
    first_val = True

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        run[f"{net_name}/learning_rate"].log(optimizer.param_groups[0]['lr'])

        t_mat = torch.zeros(2, 2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)

        for series in train_loader:
            ind, tar, chan = series
            y_pred = model(ind)
            model.zero_grad()
            pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(pred, target)
            if first_train:
                run[f"{net_name}/train_loss_pr_file"].log(loss)
                first_train = False
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
            train_acc = torch.cat((train_acc, acc.view(1)))
            t_mat = t_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

        run[f"{net_name}/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))
        train_loss = []

        run[f"{net_name}/train_acc_pr_file"].log(torch.mean(train_acc))
        train_acc = torch.tensor([]).to(device)

        run[f"{net_name}/matrix/train_confusion_matrix_pr_file"].log(t_mat)
        Accuarcy_upload(run, t_mat, total_pos, total_neg, f"{net_name}", "train")

        v_mat = torch.zeros(2,2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)


        for series in val_loader:
            ind, tar, chan = series
            y_pred = model(ind)
            pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(pred, target)
            if first_val:
                run[f"{net_name}/validation_loss_pr_file"].log(loss)
                first_val = False
            valid_loss.append(loss.item())

            acc, mat, tot_p_g, tot_n_g = Accuarcy_find(y_pred, tar, device)
            valid_acc = torch.cat((valid_acc, acc.view(1)))
            v_mat = v_mat + mat
            total_pos = total_pos + tot_p_g
            total_neg = total_neg + tot_n_g

        run[f"{net_name}/validation_loss_pr_file"].log(
                                                  np.mean(np.array(valid_loss)))
        valid_loss = []

        run[f"{net_name}/val_acc_pr_file"].log(torch.mean(valid_acc))
        valid_acc = torch.tensor([]).to(device)

        run[f"{net_name}/matrix/val_confusion_matrix_pr_file"].log(v_mat)
        Accuarcy_upload(run, v_mat, total_pos, total_neg, f"{net_name}", "val")
        if scheduler:
            scheduler.step()

    torch.save(model.state_dict(), path + f"{net_name}.pt")
    run.stop()
