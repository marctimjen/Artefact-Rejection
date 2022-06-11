import sys
sys.path.append("..") # adds higher directory to python modules path
import numpy as np
import torch
from LoaderPACK.Loader import load_whole_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def conf_mat_finder(pred, tar, classes=4):
    tar = tar.reshape(-1).numpy()
    art_pred = pred.reshape(-1).numpy()

    mat = confusion_matrix(tar, art_pred, labels = [i for i in range(classes)])
    return mat



def confusion_mat(ind: list, input_path: str, input_name:str, target_path: str,
                   target_name: str, lab_enc: dict, classes = 4, cl_for_f1=3):

    train_load_file = load_whole_data(ind=ind,
                                      input_path=input_path,
                                      input_name=input_name,
                                      target_path=target_path,
                                      target_name=target_name,
                                      input_only=False)

    train_file_loader = torch.utils.data.DataLoader(train_load_file,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)

    res_mat = np.zeros((classes, classes))

    for file in train_file_loader:
        anno = file[0][0, :, 30*200:]
        target = file[1][0, :, 30*200:]

        mat = conf_mat_finder(anno, target, classes)
        res_mat += mat

    print()
    print("Total amount of guesses:")
    print(res_mat)
    print()

    recall_mat = res_mat.copy()
    pre_mat = np.transpose(res_mat.copy())

    for i in range(len(recall_mat)):
        if (w:=recall_mat[i].sum())==0:
            continue

        recall_mat[i] = recall_mat[i]/w


    for i in range(len(pre_mat)):
        if (w:=pre_mat[i].sum())==0:
            continue

        pre_mat[i] = pre_mat[i]/w

    pre_mat = np.transpose(pre_mat)

    print("Recall confusion matrix:")
    print(recall_mat)

    print()
    print("Precision confusion matrix:")
    print(pre_mat)


    global_recall = np.sum(np.diagonal(recall_mat))/cl_for_f1
    global_precision = np.sum(np.diagonal(pre_mat))/cl_for_f1

    print("Macro-Average F1 score:", 2 * global_precision * global_recall / (global_precision + global_recall))

    labels = np.array([])

    for ke in sorted(lab_enc.keys()):
        labels = np.append(labels, [lab_enc[ke]])

    # get the predictions
    disp = ConfusionMatrixDisplay(res_mat, display_labels=labels)
    disp.plot()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # confusion matrix for recall:
    re_mat = ConfusionMatrixDisplay(recall_mat,
                                    display_labels=labels)
    re_mat.plot(ax=ax1)
    ax1.set_title("Confusion recall matrix")

    # confusion matrix for precision:
    pre_mat = ConfusionMatrixDisplay(pre_mat,
                                     display_labels=labels)
    pre_mat.plot(ax=ax2)
    ax2.set_title("Confusion precision matrix")

    plt.show()

    return res_mat
