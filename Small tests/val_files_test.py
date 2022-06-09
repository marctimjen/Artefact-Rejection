from LoaderPACK.valtester_files import val_files_test


labels = {0: "no_artifact", 1: "all_artifact", 2: "eyem", 3: "musc", 4: "elec"}

ind = [i for i in range(1, 28+1)]


# val_files_test(ind=ind,
#                input_path=r"C:\Users\Marc\Desktop\val_res_linear",
#                input_name='model_annotation ',
#                target_path=r"C:\Users\Marc\Desktop\model_data_mclass\val_model_data",
#                target_name='model_target ',
#                lab_enc=labels,
#                elec=True)

val_files_test(ind=ind,
               input_path=r"C:\Users\Marc\Desktop\val_elec_sgd_res", # sgd 634 as epoch 30
               input_name='model_annotation ',
               target_path=r"C:\Users\Marc\Desktop\model_data_mclass\val_model_data",
               target_name='model_target ',
               lab_enc=labels,
               elec=True)

# val_files_test(ind=ind,
#                input_path=r"C:\Users\Marc\Desktop\val_res_sgd_final",
#                input_name='model_annotation ',
#                target_path=r"C:\Users\Marc\Desktop\model_data_mclass\val_model_data",
#                target_name='model_target ',
#                lab_enc=labels)

# val_files_test(ind=ind,
#                input_path=r"C:\Users\Marc\Desktop\val_res_adam",
#                input_name='model_annotation ',
#                target_path=r"C:\Users\Marc\Desktop\model_data_mclass\val_model_data",
#                target_name='model_target ',
#                lab_enc=labels)