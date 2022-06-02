import torch
import numpy as np
import sys
sys.path.append("../..") # adds higher directory to python modules path

from LoaderPACK.Unet import Unet
from LoaderPACK.Loader import load_whole_data, load_5_min_intervals

load_file = load_whole_data(
            ind=range(1, 195 + 1),
            input_path="C:/Users/Marc/Desktop/model_data/train_model_data",
            input_name="model_target ",
            input_only=True) # only loads target picuters

# (self, ind: list, input_path: str, input_name = "model_input ",
#                  target_path = None, target_name = "model_target ",
#                  input_only = False, series_dict = None, series_dict_path = "")

file_loader = torch.utils.data.DataLoader(load_file,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0)

total_obs = 0
total_elec = 0
total_art = 0

for file in load_file:
    file = file[0][..., 30*200:] # remove the first 30 secs of each row
    total_obs += file.shape[0]*file.shape[1]
    total_art += np.sum(file.numpy() == 1)
    total_elec += np.sum(file.numpy() == 2)


print("Total number of observations:", total_obs)

print()

print("Total number of electrode artefacts:", total_elec)
print("Percentage of observations being an electrode artefact", total_elec/total_obs)

print()

print("Total number of other artefacts:", total_art)
print("Percentage of observations being an other artefact", total_art/total_obs)