# Artifact-Annotaion

This repository contain material for the bachelor project about artifact annotaion.
The usage of the different files/folders are:

/binary_net: This folder contain material to hyper optimize and train the binary networks.
- /1. hyperoptimize: To optimize the hyper parameters.
- /2. training networks: Used to train the final hyper-optimized network.
- /plotting: Create the plots to find the hyper-parameters. 

/LoaderPACK: Contain functions used in the training and evaluation of the networks.
- /__init__: To make this directory a python package.
- /Accuarcy_finder: Functions used in training and evalutation.
- /Accuarcy_upload: Special functions for oploading values to Neptune.
- /confusion_mat: Used to create the final test confusion matrices.
- /Loader: This script contain datasets used in the dataloading (used for training and evaluation).
- /LSTM_net: A LSTM network.
- /naive_models: Contain different naive models for artifact annotation.
- /pipeline: Functions to annotate EEG-files.
- /tester: Used to evalutate the models.
- /trainer: Function used in the final training of the networks.
- /Unet: A implementation of only the u-net structure.
- /Unet_leaky: Contain the implementation of the final network.
- /valtester_files: Function to evaluate the models predictions. This is used to find which artifacts the model is good 
at predicting.

/multiclass_net: This folder contain material to hyper optimize and train the multi-class networks.
- /1. hyperoptimize: To optimize the hyper parameters.
- /2. training networks: Used to train the final hyper-optimized network.
- /plotting: Create the plots to find the hyper-parameters. 

/Older functions & test files: Contain old files, that has been used test/make the files in this repo. 

/plotting of artifacts: To make plots of the artifacts.
- artiplot: Used to create plots of artifacts.
- range plot: Used to plot the EEG data with the signal above 199.8 µV.

/Preprocess: Used to make the preprocessing of the data.
- /0 Transform... : Script for turning the .edf and .csv file into input and target data.
- /1 train-test-split: Script to split the data into train/validation and test sets.
- /2 series_dict: Script to create series dictionary about the data.
- /3 memmap_creator: Script to make numpy memory map containing the input and target data.
- /4 memmap_copy: Move the numpy memory maps.

/Small tests: Directory of files for testing networks/functions.

/Testing the models: Used to find accuarcy and confusion matrices for the models
- /binary adam unet: Used to test the binary model utilizing the Adam optimizer.
- /binary linear model: The testing of the naive model.
- /binary sgd unet: Used to test the binary model utilizing the SGD optimizer.
- /elec sgd unet: Used to test the multi-class model utilizing the SGD optimizer.