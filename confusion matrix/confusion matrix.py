import sys
sys.path.append("..") # adds higher directory to python modules path
from LoaderPACK.confusion_mat import confusion_mat


labels = {0: "no_art", 1: "all_art"}

ind = [i for i in range(1, 28+1)]


# confusion_mat(ind=ind,
#                input_path=r"C:\Users\Marc\Desktop\val_res_linear",
#                input_name='model_annotation ',
#                target_path=r"C:\Users\Marc\Desktop\model_data_mclass\val_model_data",
#                target_name='model_target ',
#                lab_enc=labels,
#                classes=2)

confusion_mat(ind=ind,
               input_path=r"C:\Users\Marc\Desktop\val_res_adam",
               input_name='model_annotation ',
               target_path=r"C:\Users\Marc\Desktop\model_data_mclass\val_model_data",
               target_name='model_target ',
               lab_enc=labels,
               classes=2)


# labels = {0: "no_art", 1: "all_art", 2: "elec", 3:"extra"}
#
# ind = [i for i in range(1, 28+1)]
#
# confusion_mat(ind=ind,
#                 input_path=r"C:\Users\Marc\Desktop\val_elec_sgd_res", # sgd 634 as epoch 30
#                 input_name='model_annotation ',
#                 target_path=r"C:\Users\Marc\Desktop\model_data\val_model_data",
#                 target_name='model_target ',
#                 lab_enc=labels,
#                 classes=4)