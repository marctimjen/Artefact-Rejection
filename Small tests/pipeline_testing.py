import sys
sys.path.append("..") # adds higher directory to python modules path
from LoaderPACK.LSTM_net import LSTM_net
from LoaderPACK.naive_models import thenos, linear_model
from LoaderPACK.tester import val_tester
from LoaderPACK.Unet_leaky import Unet_leaky_lstm
import torch.nn as nn
import torch




batch_size = 1

device = "cpu"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# print(device)
#
# lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
#                                reduction = "mean")
#
# model = LSTM_net(batch_size=batch_size, device=device).to(device)
# x = torch.load(f"C:/Users/Marc/Desktop/network/network_LSTM-510.pt")
# # x = torch.load(r"C:\Users\Marc\Desktop\network\networks_515\network_LSTM-loss-epk-94.pt")
# model.load_state_dict(x)
# model.to(device)
#
#
# val_tester(run = "1", network = "2", model = model, lossFunc = lossFunc, device = device)

device = "cpu"

model = Unet_leaky_lstm(n_channels=1, batch_size=batch_size, \
                        device=device).to(device)

x = torch.load(f"C:/Users/Marc/Desktop/network/network/network_ADAM-470.pt", map_location='cpu')

# x = torch.load(r"C:\Users\Marc\Desktop\network\network\networks_525\network_SGD-acc-epk-21.pt", map_location='cpu')

model.load_state_dict(x)
model.to(device)

lossFunc = nn.CrossEntropyLoss(weight = torch.tensor([1., 5.]).to(device),
                               reduction = "mean")



# thenos, linear_model, model

val_tester(run = "1", network = "2", model = model, lossFunc = lossFunc,
           path = "C:/Users/Marc/Desktop/data_mclass/val_model_data", device = device)


# Older paths