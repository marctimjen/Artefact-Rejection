import sys
sys.path.append("..") # adds higher directory to python modules path
from LoaderPACK.pipeline import pipeline
from LoaderPACK.LSTM_net import LSTM_net, LSTM_net_old
from LoaderPACK.naive_models import thenos, linear_model
from LoaderPACK.Unet_leaky import Unet_leaky_lstm
import torch



# thenos, linear_model, model

device = "cpu"

batch_size = 1

model = Unet_leaky_lstm(n_channels=1, batch_size=batch_size, device=device).to(device)

x = torch.load(f"C:/Users/Marc/Desktop/network/network/network_ADAM-470.pt", map_location='cpu')

model.load_state_dict(x)
model.to(device)

pipeline(input_path=r"C:\Users\Marc\Desktop\model_data\val_model_data",
         input_name="model_input ",
         save_loc=r"C:\Users\Marc\Desktop\val_res_adam",
         ind=[i for i in range(1, 28+1)], model=model)
