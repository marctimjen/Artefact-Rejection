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
import matplotlib.pyplot as plt

def val_tester(run, network, model, lossFunc, device):
    """
    This function is used to test the accuarcy of the network.
    """

    if device == "cpu":
        fl = torch.FloatTensor
        it = torch.LongTensor
    else:
        fl = torch.cuda.FloatTensor
        it = torch.cuda.LongTensor

    load_file = testload_5min(path = "C:/Users/Marc/Desktop/data_cutoff/val_model_data",
                              series_dict = 'val_series_length.pickle',
                              size = (28, 22, 549200),
                              device = device)

    loader = torch.utils.data.DataLoader(load_file,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=0)
    valid_loss = []
    valid_acc = torch.tensor([]).to(device)
    tot_sr_nr = 0
    for series in loader:
        ind, tar, meta = series

        with torch.no_grad():
            y_pred = model(ind)

        if meta[3]:
            tar = tar[:meta[2]]
            y_pred = y_pred[:meta[2]]

        pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
        target = tar.view(-1).type(it)
        loss = lossFunc(pred, target)

        acc, mat, tot_p_g, tot_n_g, art_pred = Accuarcy_find_tester(y_pred, tar, device)

        valid_loss.append(loss)
        valid_acc = torch.cat((valid_acc, acc.view(1)))

        if acc < 0.05: # or acc > 0.95
            figure, axis = plt.subplots(2, 1)

            # The artifacts on channel: FP1-F7
            axis[0].plot(ind[0][0].cpu())
            axis[0].axes.xaxis.set_visible(False) # remove digits on x-axis

            # The artifacts on channel: F7-T3
            axis[1].plot(1*art_pred[0].cpu(), "b")
            axis[1].plot(tar[0][0].cpu(), "r", markersize=1)
            axis[1].set_title(f"Nr:{meta[0]} with channel: {meta[1]} at {meta[2]}")
            axis[1].axes.xaxis.set_visible(False) # remove digits on x-axis

            plt.savefig(f"C:/Users/Marc/Desktop/model_plots/graf ({tot_sr_nr}).jpg")
            plt.close()
            tot_sr_nr += 1

    print("mean accuarcy:", torch.mean(valid_acc))
    print("mean loss:", np.mean(valid_loss))
