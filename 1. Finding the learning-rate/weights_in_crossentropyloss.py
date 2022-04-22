import torch
import numpy as np
import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Unet import Unet
from LoaderPACK.Loader import load_whole_data, load_5_min_intervals

load_file = load_whole_data(
            path = "C:/Users/Marc/Desktop/model_data/train_model_data",
            ind = range(1, 195 + 1))

file_loader = torch.utils.data.DataLoader(load_file,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=0)

total_obs = 0
total_art = 0

for file in load_file:
    file = file[1][..., 30*200:] # remove the first 30 secs of each row
    total_obs += file.shape[0]*file.shape[1]
    total_art += np.sum(file.numpy())


print("Total number of observations:", total_obs)
print("Total number of artefacts:", total_art)
print("Percentage of observations being an artefact", total_art/total_obs)
