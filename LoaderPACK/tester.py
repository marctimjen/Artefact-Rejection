import numpy as np
import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Loader import testload_5min
from LoaderPACK.Accuarcy_finder import Accuarcy_find_tester, recall_find_tester, histogram_find_tester
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def val_tester(run, network, model, lossFunc, path, device):
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

    load_file = testload_5min(path = path,
                              series_dict = 'val_series_length.pickle',
                              size = (28, 22, 549200), # size = (195, 22, 2060000),
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

    samples = torch.tensor([]).to(device)

    matrix = torch.zeros(2, 2)

    thrs =.5

    number_bins = 1000
    hist_p_guess = np.zeros(number_bins)
    hist_n_guess = np.zeros(number_bins)

    precision_tp = torch.tensor([]).to(device)
    precision_tn = torch.tensor([]).to(device)

    recall_tp = torch.tensor([]).to(device)
    recall_tn = torch.tensor([]).to(device)

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

        # samples = torch.cat((samples, torch.Tensor([y_pred.shape[2]])))

        # pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)
        # target = tar.view(-1).type(it)
        # loss = lossFunc(pred, target)

        acc, mat, tot_p_g, tot_n_g, art_pred = Accuarcy_find_tester(y_pred, tar, thrs)

        matrix = matrix + mat

        # valid_loss.append(loss.item())
        # valid_acc = torch.cat((valid_acc, acc.view(1)))

        # precision_tp = torch.cat((precision_tp, (mat[0][0]/tot_p_g).view(1)))
        # precision_tn = torch.cat((precision_tn, (mat[1][1]/tot_n_g).view(1)))

        # r_tp, r_tn = recall_find_tester(mat)

        # recall_tp = torch.cat((recall_tp, r_tp.view(1)))
        # recall_tn = torch.cat((recall_tn, r_tn.view(1)))

        p_guess, n_guess = histogram_find_tester(y_pred, tar)

        # roc_tar = np.concatenate((roc_tar, target.numpy()))
        # roc_pred = np.concatenate((roc_pred, y_pred.view(2, -1)[1].numpy()))

        # if len(roc_pred) >= 6000000:
        #     print("yo")
        #     roc_s.append(roc_auc_score(roc_tar, roc_pred))
        #     roc_pred = np.array([])
        #     roc_tar = np.array([])


        if p_guess.nelement():
            first, second = np.histogram(p_guess.numpy(), bins=np.arange(number_bins + 1) / number_bins)
            hist_p_guess += first

        if n_guess.nelement():
            first, second = np.histogram(n_guess.numpy(), bins=np.arange(number_bins + 1) / number_bins)

            # for i in first:
            #     if first[-1] > 100:
            #         pass

            hist_n_guess += first

        if acc > -0.05: #np.random.choice([0, 1], p = [0.99, 0.01]): # # or acc > 0.95:
            figure, axis = plt.subplots(2, 1)

            # The artifacts on channel: FP1-F7
            axis[0].plot(ind[0][0].cpu())
            axis[0].set_title(f"Input EEG data")
            axis[0].axes.xaxis.set_visible(False) # remove digits on x-axis

            start_point = meta[2].numpy()[0]

            min_t1 = int((start_point/200)//60)
            sec_t1 = int(start_point/200 % 60)
            min_t2 = int(min_t1 + 5)
            # The artifacts on channel: F7-T3
            axis[1].plot(1*art_pred[0].cpu(), "b", label = "annotations") # prediction is blue
            axis[1].plot(tar[0][0].cpu(), "r", markersize=1, label = "targets") # true target = red
            axis[1].set_title(f"Annotations")
            #axis[1].axes.xaxis.set_visible(False) # remove digits on x-axis
            axis[1].set_xticks(ticks=[0, 60000], labels=[f"{min_t1}:{sec_t1}", f"{min_t2}:{sec_t1}"])
            axis[1].legend()



            plt.savefig(f"C:/Users/Marc/Desktop/model_plots/graf {tot_sr_nr} (Nr {int(meta[0])} with channel {int(meta[1])}).jpg")
            plt.close()
            tot_sr_nr += 1

    # print("mean accuarcy:", (valid_acc * samples).nansum()/samples.sum())
    # print("mean true positive precision:", (precision_tp * samples).nansum()/samples.sum())
    # print("mean true negative precision:", (precision_tn * samples).nansum()/samples.sum())
    # print("mean true positive recall:", (recall_tp * samples).nansum()/samples.sum())
    # print("mean true negative recall:", (recall_tn * samples).nansum()/samples.sum())

    print("mean accuarcy:", (matrix[0][0]+matrix[1][1])/matrix.sum())
    print("mean true positive precision:", matrix[0][0]/(matrix[0][0]+matrix[0][1]))
    print("mean true negative precision:", matrix[1][1]/(matrix[1][1]+matrix[1][0]))
    print("mean true positive recall:", matrix[0][0]/(matrix[0][0]+matrix[1][0]))
    print("mean true negative recall:", matrix[1][1]/(matrix[1][1]+matrix[0][1]))


    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title('Models prediction when no artefact is present (target == 0)')
    ax1.bar(np.arange(number_bins) / number_bins, hist_n_guess, width=1/1000, color=['green' if i < int(thrs * 1000)
                                                                                     else "red" for i in
                                                                                     range(number_bins)])
    ax1.set_xlim([-0.01, 1.01])
    ax1.axvline(x=thrs, color='m', linestyle="--", label=f'Threshold = {thrs}')
    ax1.text(0.25, 0.5, 'True negative (TN)', horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, color="g",
             bbox=dict(boxstyle="round",
                       facecolor="gray",
                       alpha=0.3))

    ax1.text(0.75, 0.5, 'False negative (FN)', horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes, color="r",
             bbox=dict(boxstyle="round",
                       facecolor="gray",
                       alpha=0.3))

    ax2.set_title('Models prediction when an artefact is present (target != 0)')
    ax2.bar(np.arange(number_bins) / number_bins, hist_p_guess, width=1/1000, color=['red' if i < int(thrs * 1000)
                                                                                     else "green" for i in
                                                                                     range(number_bins)])
    ax2.set_xlim([-0.01, 1.01])
    ax2.axvline(x=thrs, color='m', linestyle="--", label=f'Threshold = {thrs}')
    ax2.text(0.25, 0.5, 'False positive (FP)', horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, color="r",
             bbox=dict(boxstyle="round",
                       facecolor="gray",
                       alpha=0.3))

    ax2.text(0.75, 0.5, 'True positive (TP)', horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes, color="g",
             bbox=dict(boxstyle="round",
                       facecolor="gray",
                       alpha=0.3))
    plt.show()


    # print("mean loss:",  np.mean(valid_loss))
    # print("roc", np.mean(np.array(roc_s)))
