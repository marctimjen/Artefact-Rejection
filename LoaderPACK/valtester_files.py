import sys
sys.path.append("..") # adds higher directory to python modules path

import torch
from LoaderPACK.Loader import load_whole_data, load_5_min_intervals
from LoaderPACK.Accuarcy_finder import mclass_acc_recal_fidner, recall_find_tester, single_mclass_acc_recal_finder



def val_files_test(ind: list, input_path: str, input_name:str, target_path: str,
                   target_name: str, lab_enc: dict, device):

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

    class_dict = dict()
    for i in lab_enc.keys():
        class_dict[i] = {
            "mat": torch.zeros(2, 2) # matrix
        }

    for file in train_file_loader:
        anno = file[0][0, :, 30*200:]
        target = file[1][0, :, 30*200:]
        it = iter(single_mclass_acc_recal_finder(anno, target))

        for (acc, mat, tot_p_g, tot_n_g, art_pred, i) in it:

            in_dict = class_dict[i] # get the inner dictionary

            in_dict["mat"] += mat



    for i in lab_enc.keys():
        mat = class_dict[i]["mat"]
        r_tp, r_tn = recall_find_tester(mat)  # find the recall
        print("Artifact type:", lab_enc[i])
        print()
        print("mean accuarcy:", (mat[0][0]+mat[1][1])/mat.sum())
        print("mean true positive precision:", mat[0][0]/(mat[0][0]+mat[0][1]))
        print("mean true negative precision:", mat[1][1]/(mat[1][0]+mat[1][1]))
        print("mean true positive recall:", r_tp)
        print("mean true negative recall:", r_tn)
        print("------------------------------------")
