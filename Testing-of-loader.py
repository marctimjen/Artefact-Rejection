import torch
from torch.utils.data import Dataset, DataLoader
from LoaderPACK.Loader import load_whole_data, load_shuffle_5_min


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

trainload = load_whole_data(path = "C:/Users/Marc/Desktop/model_data", ind = [i for i in range(1, 310 + 1)])
batch_size = 1

# Set up the dataloaders:

trainloader = torch.utils.data.DataLoader(trainload,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)
load = iter(trainloader)

loader2 = load_shuffle_5_min(next(load), "cpu")
batch_size = 1

# Set up the dataloaders:

loader2 = torch.utils.data.DataLoader(loader2,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

load2 = iter(loader2)

j = 0

for i in load2:
    print(i)
    print(i[0].shape)
    j += 1

print(j)
