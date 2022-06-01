import sys
sys.path.append("..") # adds higher directory to python modules path

import torch
from LoaderPACK.Loader import load_whole_data, load_5_min_intervals
from LoaderPACK.Accuarcy_finder import mclass_acc_recal_fidner



def val_files_test(ind: list, input_path: str, input_name:str, target_path: str, target_name: str):

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
        anno = file[0]
        target = file[1]
        it = iter(mclass_acc_recal_fidner(anno, target))

        for ll in it:
            pass

