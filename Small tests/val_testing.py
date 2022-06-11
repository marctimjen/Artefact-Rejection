import sys
sys.path.append("..") # adds higher directory to python modules path
from LoaderPACK.LSTM_net import LSTM_net, LSTM_net_old
from LoaderPACK.naive_models import thenos, linear_model
from LoaderPACK.tester import val_tester
from LoaderPACK.Unet_leaky import Unet_leaky_lstm
import torch.nn as nn
import torch


batch_size = 1

device = "cpu"


model = Unet_leaky_lstm(n_channels=1, batch_size=batch_size, device=device).to(device)

x = torch.load(f"C:/Users/Marc/Desktop/network/network/network_ADAM-470.pt", map_location='cpu')
# x = torch.load(r"C:\Users\Marc\Desktop\network\network\networks_525\network_SGD-acc-epk-21.pt", map_location='cpu')
model.load_state_dict(x)
model.to(device)

lossFunc = nn.CrossEntropyLoss(weight=torch.tensor([1., 5.]).to(device),
                               reduction="mean")

# linear_model, model

# val_tester(run="1", network="2", model=linear_model, lossFunc=lossFunc,
#            path="C:/Users/Marc/Desktop/data_mclass/val_model_data", device=device)

val_tester(run="1", network="2", model=model, lossFunc=lossFunc,
           path="C:/Users/Marc/Desktop/data_mclass/val_model_data", device=device)
