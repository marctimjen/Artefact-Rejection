# This script is for copying some files to another folder

import shutil

def mover(path_from: str, path_to: str, file: str):
    """
    This function moves the file from the directory of path_from to the
    directory of path_to.
    """

    shutil.copyfile(path_from + "/" + file, path_to + "/" + file)


if __name__ == "__main__":
    path_from = "C:/Users/Marc/Desktop/model_data/train_model_data"
    path_to = "C:/Users/Marc/Desktop/data/train_model_data"

    mover(path_from, path_to, file = "model_target.dat")
    mover(path_from, path_to, file = "model_input.dat")
    mover(path_from, path_to, file = "train_encoding.csv")
    mover(path_from, path_to, file = "train_series_length.pickle")

    path_from = "C:/Users/Marc/Desktop/model_data/val_model_data"
    path_to = "C:/Users/Marc/Desktop/data/val_model_data"

    mover(path_from, path_to, file = "model_target.dat")
    mover(path_from, path_to, file = "model_input.dat")
    mover(path_from, path_to, file = "val_encoding.csv")
    mover(path_from, path_to, file = "val_series_length.pickle")
