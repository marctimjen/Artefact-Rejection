import mne
import csv
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch

dir1_eeg = "C:/Users/Bruger/Documents/Uni/6. Semester/BP/data/v2.1.0/lists/edf_01_tcp_ar.list"
dir2_eeg = "C:/Users/Bruger/Documents/Uni/6. Semester/BP/data/v2.1.0/lists/edf_02_tcp_ar.list"
dir3_eeg = "C:/Users/Bruger/Documents/Uni/6. Semester/BP/data/v2.1.0/lists/edf_03_tcp_ar.list"

dir1_rec = "C:/Users/Bruger/Documents/Uni/6. Semester/BP/data/v2.1.0/lists/rec_01_tcp_ar.list"
dir2_rec = "C:/Users/Bruger/Documents/Uni/6. Semester/BP/data/v2.1.0/lists/rec_02_tcp_ar.list"
dir3_rec = "C:/Users/Bruger/Documents/Uni/6. Semester/BP/data/v2.1.0/lists/rec_03_tcp_ar.list"

class data_loader(Dataset):
    def __init__(self):
        self.eeg_path = dir1_eeg
        # Path til billederne + csv fil med labels
        self.data = []
        with open(dir1_eeg, "r") as file: # Læser csv-fil
            list = csv.reader(file)
            for rows in list:
                self.data.append(rows)

        length = len(self.data)

        with open(dir1_rec, "r") as file: # Læser csv-fil
            list = csv.reader(file)
            for i in range(length):
                self.data[i].append(next(list)[0])
        # Gemmer dataet i self.data. Formatet er [[dir til edf, dir til rec], ...]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_data = mne.io.read_raw_edf(self.data[idx][0])

        target = []
        with open(self.data[idx][1], "r") as file: # Læser csv-fil
            list = csv.reader(file)
            for rows in list:
                target.append([int(rows[0]), float(rows[1]), float(rows[2]), int(rows[3])])

        print(target)

        return eeg_data, target


batch_size = 1
# Set up the dataset.
dataset = data_loader()

# Set up the dataset.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)
# get some images
dataiter = iter(trainloader)
data = dataiter.next()

print(data)
