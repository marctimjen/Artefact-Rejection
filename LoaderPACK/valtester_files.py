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
            "acc": torch.tensor([]).to(device), # accuarcy measured by precision
            "pre_tp": torch.tensor([]).to(device), # precision true positive
            "pre_tn": torch.tensor([]).to(device), # precision true negative
            "rec_tp": torch.tensor([]).to(device), # recall true positive
            "rec_tn": torch.tensor([]).to(device)  # recall true negative
        }

    for file in train_file_loader:
        anno = file[0]
        target = file[1]
        it = iter(single_mclass_acc_recal_finder(anno, target))

        for (acc, mat, tot_p_g, tot_n_g, art_pred, i) in it:

            r_tp, r_tn = recall_find_tester(mat) # find the recall

            in_dict = class_dict[i] # get the inner dictionary

            in_dict["acc"] = torch.cat((in_dict["acc"], acc.view(1))) # accuarcy overall

            in_dict["pre_tp"] = torch.cat((in_dict["pre_tp"], (mat[0][0] / tot_p_g).view(1)))
                # precision tp

            in_dict["pre_tn"] = torch.cat((in_dict["pre_tn"], (mat[1][1] / tot_n_g).view(1)))
                # precision tn

            in_dict["rec_tp"] = torch.cat((in_dict["rec_tp"], r_tp.view(1))) # recall tp
            in_dict["rec_tn"] = torch.cat((in_dict["rec_tn"], r_tn.view(1))) # recall tn


    for i in lab_enc.keys():
        print()
        print("Artifact type:", lab_enc[i])
        print()
        print("mean accuarcy:", torch.nanmean(class_dict[i]["acc"]))
        print("mean true positive precision:", torch.nanmean(class_dict[i]["pre_tp"]))
        print("mean true negative precision:", torch.nanmean(class_dict[i]["pre_tn"]))
        print("mean true positive recall:", torch.nanmean(class_dict[i]["rec_tp"]))
        print("mean true negative recall:", torch.nanmean(class_dict[i]["rec_tn"]))
        print("------------------------------------")
