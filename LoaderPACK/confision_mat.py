import sys
sys.path.append("..") # adds higher directory to python modules path

import torch
from LoaderPACK.Loader import load_whole_data, load_5_min_intervals
from sklearn.metrics import confusion_matrix

def conf_mat_finder(pred, tar, labels, classes=3):
    tar = tar.reshape(-1).numpy()
    art_pred = pred.reshape(-1).numpy()

    mat = confusion_matrix(tar, art_pred)

    return mat



def val_files_test(ind: list, input_path: str, input_name:str, target_path: str,
                   target_name: str, lab_enc: dict, classes = 3):

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


    for file in train_file_loader:
        anno = file[0][0, :, 30*200:]
        target = file[1][0, :, 30*200:]

        mat = conf_mat_finder(anno, target, lab_enc)
