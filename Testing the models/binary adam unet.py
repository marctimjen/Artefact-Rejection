# this script is for running the binary unet of adam

import sys
sys.path.append("..") # adds higher directory to python modules path
from LoaderPACK.pipeline import pipeline
from LoaderPACK.Unet_leaky import Unet_leaky_lstm
from LoaderPACK.valtester_files import val_files_test
from LoaderPACK.confusion_mat import confusion_mat
import torch

device = "cpu"
batch_size = 1

model = Unet_leaky_lstm(n_channels=1, batch_size=batch_size, device=device).to(device)
x = torch.load(f"C:/Users/Marc/Desktop/network/network/network_ADAM-470.pt", map_location='cpu')

model.load_state_dict(x)
model.to(device)

ind = [i for i in range(1, 56+1)]


# get the predictions from the adam
# pipeline(input_path=r"C:\Users\Marc\Desktop\model_data\test_model_data",
#          input_name="model_input ",
#          save_loc=r"C:\Users\Marc\Desktop\test_adam_binary",
#          ind=ind, model=model)


labels = {0: "no_artifact", 1: "all_artifact", 2: "elec", 3: "musc", 4: "eyem"}

# get the which annotations are correct
val_files_test(ind=ind,
               input_path=r"C:\Users\Marc\Desktop\test_adam_binary",
               input_name='model_annotation ',
               target_path=r"C:\Users\Marc\Desktop\model_data_mclass\test_model_data",
               target_name='model_target ',
               lab_enc=labels,
               elec=False)

# finally get the confusion matrix:
labels = {0: "clean", 1: "artifact"} # only binary input

confusion_mat(ind=ind,
                input_path=r"C:\Users\Marc\Desktop\test_adam_binary",
                input_name='model_annotation ',
                target_path=r"C:\Users\Marc\Desktop\model_data\test_model_data",
                target_name='model_target ',
                lab_enc=labels,
                classes=2,
                cl_for_f1=2)


# for the validation set:

# ind = [i for i in range(1, 28+1)]
# val_files_test(ind=ind,
#                input_path=r"C:\Users\Marc\Desktop\val_res_adam",
#                input_name='model_annotation ',
#                target_path=r"C:\Users\Marc\Desktop\model_data_mclass\val_model_data",
#                target_name='model_target ',
#                lab_enc=labels,
#                elec=False)
