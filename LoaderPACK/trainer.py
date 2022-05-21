import neptune.new as neptune
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.multiprocessing as mp
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

    valid_loss, train_loss = [], []
    valid_acc = torch.tensor([]).to(device)
    train_acc = torch.tensor([]).to(device)

    first_loss_save = True

    try: # test if the optimizor contain momentum
        moment = optimizer.param_groups[0]['momentum']
        moment = True
    except:
        moment = False

    for iEpoch in range(nEpoch):
        print(f"Training epoch {iEpoch}")

        run[f"{net_name}/learning_rate"].log(optimizer.param_groups[0]['lr'])

        if moment:
            run[f"{net_name}/momentum"].log(
                                          optimizer.param_groups[0]['momentum'])

        t_mat = torch.zeros(2, 2)
        total_pos, total_neg = torch.tensor(0), torch.tensor(0)

        for series in train_loader:
            ind, tar, chan = series
            y_pred = model(ind)
            model.zero_grad()
            pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
            target = tar.view(-1).type(it)
            loss = lossFunc(pred, target)
            if first_loss_save:
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

        run[f"{net_name}/train_loss_pr_file"].log(
                                                np.mean(np.array(train_loss)))

        sm_loss = np.mean(np.array(train_loss)) * smooth \
                            + (1-smooth) * t_sm_loss

        t_sm_loss = sm_loss
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

        run[f"{net_name}/validation_loss_pr_file"].log(
                                                  np.mean(np.array(valid_loss)))

        sm_loss = np.mean(np.array(valid_loss)) * smooth \
                            + (1-smooth) * v_sm_loss

        v_sm_loss = sm_loss

        run[f"{net_name}/smooth_val_loss_pr_file"].log(sm_loss)
        valid_loss = []

        run[f"{net_name}/val_acc_pr_file"].log(torch.mean(valid_acc))
        valid_acc = torch.tensor([]).to(device)

        run[f"{net_name}/matrix/val_confusion_matrix_pr_file"].log(v_mat)
        Accuarcy_upload(run, v_mat, total_pos, total_neg, f"{net_name}", "val")

    str_run = run.get_run_url()
    m = re.match(r".+-(\d+)", str_run) # this correlates the name of the network
                                  # with the neptune ai name.

    torch.save(model.state_dict(), path + f"{net_name}-{(m.group(1))}.pt")
    run.stop()
