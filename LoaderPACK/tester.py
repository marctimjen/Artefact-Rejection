import csv
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Loader import testload_5min
from LoaderPACK.Accuarcy_finder import Accuarcy_find_tester
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def val_tester(run, network, model, lossFunc, device):
    """
    This function is used to test the accuarcy of the network..
    """

    if device == "cpu":
        fl = torch.FloatTensor
        it = torch.LongTensor
    else:
        fl = torch.cuda.FloatTensor
        it = torch.cuda.LongTensor

        # "C:/Users/Marc/Desktop/data/val_model_data"
        # "C:/Users/Marc/Desktop/data/train_model_data"

    load_file = testload_5min(path = "C:/Users/Marc/Desktop/data/val_model_data",
                              series_dict = 'val_series_length.pickle',
                              size = (28, 22, 549200), #size = (195, 22, 2060000),
                              device = device) # total val series = 2642
                                               # total train series = 18497

    loader = torch.utils.data.DataLoader(load_file,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0)
    valid_loss = []
    valid_acc = torch.tensor([]).to(device)

    roc_pred = np.array([])
    roc_tar = np.array([])

    roc_s = []

    t_p_rate = torch.tensor([]).to(device)
    t_n_rate = torch.tensor([]).to(device)

    tot_sr_nr = 0
    counter = -1
    for series in loader:
        counter += 1
        if not(counter % 100):
            print(counter)

        ind, tar, meta = series


        with torch.no_grad():
            y_pred = model(ind)

        if meta[3]:


            tar = tar[0][0][:meta[4]].view(1, 1, -1)

            ind = ind[0][0][:meta[4]].view(1, 1, -1)

            a = y_pred[0][0][:meta[4]].view(1, -1)
            b = y_pred[0][1][:meta[4]].view(1, -1)
            y_pred = torch.stack((a, b), dim = 1)

        # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
        target = tar.view(-1).type(it)
        # loss = lossFunc(pred, target)

        acc, mat, tot_p_g, tot_n_g, art_pred = Accuarcy_find_tester(y_pred, tar, device)

        # valid_loss.append(loss.item())
        valid_acc = torch.cat((valid_acc, acc.view(1)))

        t_p_rate = torch.cat((t_p_rate, (mat[0][0]/tot_p_g).view(1)))
        t_n_rate = torch.cat((t_n_rate, (mat[1][1]/tot_n_g).view(1)))

        # roc_tar = np.concatenate((roc_tar, target.numpy()))
        # roc_pred = np.concatenate((roc_pred, y_pred.view(2, -1)[1].numpy()))

        # if len(roc_pred) >= 6000000:
        #     print("yo")
        #     roc_s.append(roc_auc_score(roc_tar, roc_pred))
        #     roc_pred = np.array([])
        #     roc_tar = np.array([])


        if acc < -0.05: # or acc > 0.95:
            figure, axis = plt.subplots(2, 1)

            # The artifacts on channel: FP1-F7
            axis[0].plot(ind[0][0].cpu())
            axis[0].axes.xaxis.set_visible(False) # remove digits on x-axis

            # The artifacts on channel: F7-T3
            axis[1].plot(1*art_pred[0].cpu(), "b") # prediction is blue
            axis[1].plot(tar[0][0].cpu(), "r", markersize=1) # true target = red
            axis[1].set_title(f"Nr:{meta[0]} with channel: {meta[1]} at {meta[2]}")
            axis[1].axes.xaxis.set_visible(False) # remove digits on x-axis

            plt.savefig(f"C:/Users/Marc/Desktop/model_plots/graf ({tot_sr_nr}).jpg")
            plt.close()
            tot_sr_nr += 1

    print("mean accuarcy:", torch.nanmean(valid_acc))
    print("mean true positive rate:", torch.nanmean(t_p_rate))
    print("mean true negative rate:", torch.nanmean(t_n_rate))
    # print("mean loss:",  np.mean(valid_loss))
    # print("roc", np.mean(np.array(roc_s)))
