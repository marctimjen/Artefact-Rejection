# this script is for running the multi-class unet using SGD

import sys
sys.path.append("..") # adds higher directory to python modules path
from LoaderPACK.pipeline import pipeline
from LoaderPACK.Unet_leaky import Unet_leaky_lstm_elec
from LoaderPACK.valtester_files import val_files_test
from LoaderPACK.confusion_mat import confusion_mat
import torch

device = "cpu"
batch_size = 1

model = Unet_leaky_lstm_elec(n_channels=1, batch_size=batch_size, device=device).to(device)
x = torch.load(f"C:/Users/Marc/Desktop/sgd elec 634/network_SGD-loss-epk-30.pt", map_location='cpu')

model.load_state_dict(x)
model.to(device)

ind = [i for i in range(1, 56+1)]


# get the predictions from the sgd
# pipeline(input_path=r"C:\Users\Marc\Desktop\model_data\test_model_data",
#          input_name="model_input ",
#          save_loc=r"C:\Users\Marc\Desktop\test_sgd_elec",
#          ind=ind, model=model)


labels = {0: "no_artifact", 1: "all_artifact", 2: "elec", 3: "musc", 4: "eyem"}

# get the which annotations are correct
val_files_test(ind=ind,
               input_path=r"C:\Users\Marc\Desktop\test_sgd_elec",
               input_name='model_annotation ',
               target_path=r"C:\Users\Marc\Desktop\model_data_mclass\test_model_data",
               target_name='model_target ',
               lab_enc=labels,
               elec=True)


# finally get the confusion matrix:
labels = {0: "no_art", 1: "all_art", 2: "elec", 3: "extra"} # extra for the extra channel

confusion_mat(ind=ind,
                input_path=r"C:\Users\Marc\Desktop\test_sgd_elec",
                input_name='model_annotation ',
                target_path=r"C:\Users\Marc\Desktop\model_data_elec\test_model_data",
                target_name='model_target ',
                lab_enc=labels,
                classes=4)
