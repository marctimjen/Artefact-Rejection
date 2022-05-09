import csv
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..") # adds higher directory to python modules path

from LoaderPACK.Loader import testload_5min
import torch

val_load_file = testload_5min(path = "/home/tyson/data_cutoff/val_model_data",
                                     series_dict = 'val_series_length.pickle',
                                     size = (28, 22, 2642),
                                     device = "cpu")


val_loader = torch.utils.data.DataLoader(val_load_file,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=0)
j = 0
for i in val_loader:
    j += 1

print("Total series loaded:", j)
