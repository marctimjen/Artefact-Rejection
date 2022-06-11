# this script is for running the binary unet of adam

import sys
sys.path.append("..") # adds higher directory to python modules path
from LoaderPACK.pipeline import pipeline
from LoaderPACK.valtester_files import val_files_test
from LoaderPACK.confusion_mat import confusion_mat
from LoaderPACK.naive_models import linear_model



ind = [i for i in range(1, 56+1)]

model = linear_model

# get the predictions from the linear_model
# pipeline(input_path=r"C:\Users\Marc\Desktop\model_data\test_model_data",
#          input_name="model_input ",
#          save_loc=r"C:\Users\Marc\Desktop\test_linear_binary",
#          ind=ind, model=model)


labels = {0: "no_artifact", 1: "all_artifact", 2: "elec", 3: "musc", 4: "eyem"}

# get the which annotations are correct
val_files_test(ind=ind,
               input_path=r"C:\Users\Marc\Desktop\test_linear_binary",
               input_name='model_annotation ',
               target_path=r"C:\Users\Marc\Desktop\model_data_mclass\test_model_data",
               target_name='model_target ',
               lab_enc=labels,
               elec=False)


# finally get the confusion matrix:
labels = {0: "clean", 1: "artifact"} # only binary input

confusion_mat(ind=ind,
                input_path=r"C:\Users\Marc\Desktop\test_linear_binary",
                input_name='model_annotation ',
                target_path=r"C:\Users\Marc\Desktop\model_data\test_model_data",
                target_name='model_target ',
                lab_enc=labels,
                classes=2,
                cl_for_f1=2)



# confusion matrix for validation data set

# ind = [i for i in range(1, 28+1)]
#
# confusion_mat(ind=ind,
#                 input_path=r"C:\Users\Marc\Desktop\val_res_linear",
#                 input_name='model_annotation ',
#                 target_path=r"C:\Users\Marc\Desktop\model_data\val_model_data",
#                 target_name='model_target ',
#                 lab_enc=labels,
#                 classes=2,
#                 cl_for_f1=2)