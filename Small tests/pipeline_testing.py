import sys
sys.path.append("..") # adds higher directory to python modules path
from LoaderPACK.pipeline import pipeline
from LoaderPACK.LSTM_net import LSTM_net, LSTM_net_old
from LoaderPACK.naive_models import thenos, linear_model
from LoaderPACK.Unet_leaky import Unet_leaky_lstm
import torch



# thenos, linear_model, model

pipeline(input_path=r"C:\Users\Marc\Desktop\model_data_mclass\test_model_data",
         input_name="model_input ",
         save_loc=r"C:\Users\Marc\Desktop\test_res",
         ind=[i for i in range(1, 56+1)], model=linear_model)
