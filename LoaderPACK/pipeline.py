import sys
sys.path.append("..") # adds higher directory to python modules path

import torch
from LoaderPACK.Loader import load_whole_data, load_5_min_intervals




def pipeline(input_path: str, input_name: str, save_loc: str, ind: list, model, start=200*30):
    """
    This function is used to create annotations for the input EEG files. 
    Note: The EEG files should be on a .pt format 
    (use "0 Transform_data..." to get the EEG on this format). 

    Args:
        input_path (str): path to a file containing edf files.
        input_name (str): name of the EEG files.
        save_loc (str): path to save the final annotaions.
        ind (str): indicies (if multiple files is to be loaded).
        model: pytorch model to find the annotations.
        start (int): how many sampels to drop in the beggining of the
                     EEG-recording.

    Produced files:
        This function creates annotated files from the EEG-input. This
        is the models predictions.
    """

    device = "cpu"
    batch_size = 1


    train_load_file = load_whole_data(ind=ind, input_path=input_path, input_name=input_name,
                                      input_only=True)

    train_file_loader = torch.utils.data.DataLoader(train_load_file,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0)

    it = iter(ind)

    for file in train_file_loader:

        nr = next(it) # get the file number

        result = torch.zeros(file[0].shape[1], file[0].shape[2]) # make target data

        # the second loader is for loading the 5-mins intervals
        load_series = load_5_min_intervals(file, device, start=start)

        series_loader = torch.utils.data.DataLoader(load_series,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)
        for series in series_loader:
            ind, tar, chan, cut = series

            with torch.no_grad():
                pred = model(ind)

            y_pred = (pred[:, 1] >= 0.5).view(-1)

            cut_end = min(cut + 60 * 5 *200, result.shape[1])
            result[int(chan)][cut:cut_end] = y_pred[:cut_end-cut]

        torch.save(result, save_loc + f'/model_annotation ({nr}).pt') # save the annotations