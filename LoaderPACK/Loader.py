from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import math
import pickle

class load_whole_data(Dataset): # Dataset
    """
    This dataloader loads the tensor input and target in whole
    """
    def __init__(self, path: str, ind: list, series_dict = None):
        """
        Args:
            path (str): path to the input & target folder.
            ind (list): list of indices for which pictures to load.
            device (class 'torch.device'): which pytorch device the data should
            be sent to.
        """

        self.device = "cpu"
        self.imgs_path = path

        if series_dict:
            with open(path + "/" + series_dict, 'rb') as handle:
                self.s_dict = pickle.load(handle)
        else:
            self.s_dict = False

        self.data = []
        for i in ind:
            self.data.append([self.imgs_path + f"/model_input ({i}).pt",
                        self.imgs_path + f"/model_target ({i}).pt",
                        self.s_dict[i] if self.s_dict else 0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_path, target_path, s_len_ls = self.data[idx]
                                # path for target + input + lenght of series

        inp = torch.load(input_path) # load the input data
        inp = inp.type(torch.float).to(self.device)

        tar = torch.load(target_path) # load the target data
        tar = tar.type(torch.float).to(self.device)

        return inp, tar, s_len_ls




class load_shuffle_5_min(Dataset):
    """
    This dataloader loads the tensor input and target in whole
    """
    def __init__(self, ls: list, device):
        """
        Args:
            path (str): path to the input & target folder.
            ind (list): list of indices for which pictures to load.
            device (class 'torch.device'): which pytorch device the data should
            be sent to.
        """

        self.device = device
        self.ls = ls # list with the input and target data
        self.size = (ls[0][0].shape[0], ls[0][0].shape[1])
            # size of target and input

        self.s_len = ls[2][0]
        self.s_num = ls[2][1]

        length = math.floor((self.size[1]/(200*60*5)))*self.size[0]
            # the amount of total possible cuts

        self.length = int(min(torch.div(110, self.s_num,
                                        rounding_mode='trunc'), length))

        # self.length = int(min(torch.div((130 - self.s_len % 130), self.s_num,
        #                                 rounding_mode='trunc'), length))
            # make sure that no more than 75 samples is taken from the same
            # individual

        self.gen = iter(self.create_data(self.length))


    def create_data(self, nr_of_cuts):
        cut_point = np.random.randint(low = 200*30, #remove the first 30 secs
                            high = self.size[1] - 5*200*60, size = nr_of_cuts)
                            # choose the place to cut

        cuts_pr_chan = nr_of_cuts/self.ls[0][0].shape[0]
            # the amount of cuts pr channel

        for i in range(nr_of_cuts):
            chan = int(i//cuts_pr_chan) # the given channel
            inp = self.ls[0][0][chan][cut_point[i]:cut_point[i]+60*5*200].view(1, 60*5*200)
            tar = self.ls[1][0][chan][cut_point[i]:cut_point[i]+60*5*200].view(1, 60*5*200)
            #inp = self.ls[0][0][chan][cut_point[i]:cut_point[i]+60*5*200]
            #tar = self.ls[1][0][chan][cut_point[i]:cut_point[i]+60*5*200]


            #tar = torch.cat((tar[0], -1*(tar[0] - 1))).view(2, 60*5*200)
            yield (inp, tar, chan)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inp, tar, chan = next(self.gen)
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        return inp, tar, chan



class load_5_min_intervals(Dataset):
    """
    This dataloader loads the tensor input and target in whole
    """
    def __init__(self, ls: list, device):
        """
        Args:
            path (str): path to the input & target folder.
            ind (list): list of indices for which pictures to load.
            device (class 'torch.device'): which pytorch device the data should
            be sent to.
        """

        self.device = device
        self.ls = ls # list with the input and target data
        self.size = [ls[0][0].shape[0], ls[0][0].shape[1]]
            # size of target and input

        # zero-pad the result if it can't be in only 5 mins intervals.
        extra = 200*60*5 - (w := (ls[0][0].shape[1]-30*200) % (200*60*5))


        if w: # if w is not equal to 0, then zero-pad is needed:
            # zero pad:
            self.ls[0] = F.pad(self.ls[0], (0, extra), "constant", 0.0)
            self.ls[1] = F.pad(self.ls[1], (0, extra), "constant", 0.0)

            self.size[1] = self.size[1] + extra


        self.length = math.floor((self.size[1]-30*200)/(200*60*5))*self.size[0]
            # the amount of total possible cuts

        self.gen = iter(self.cut_data())



    def cut_data(self):
        for chan in range(self.size[1]):
            for cut_point in range(30*200, self.size[1], 200*5*60):
                inp = self.ls[0][0][chan][cut_point:cut_point+60*5*200].view(1, 60*5*200)
                tar = self.ls[1][0][chan][cut_point:cut_point+60*5*200].view(1, 60*5*200)
                yield (inp, tar, chan, cut_point)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inp, tar, chan, cut = next(self.gen)
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        return inp, tar, chan, cut



# class shuffle_5min(Dataset):
#     """
#     This dataloader loads random 5 minute intervals from a random patient.
#     """
#     def __init__(self, path: str, series_dict: str, size: tuple, device):
#         """
#         Args:
#             path (str): path to the input & target folder.
#             series_dict (list): name of dict for data.
#             size : (number of experiments, number of max. channels, longest series)
#             device (class 'torch.device'): which pytorch device the data should
#             be sent to.
#         """
#
#         self.device = device
#         self.size = size
#         self.path = path
#
#         with open(path + "/" + series_dict, 'rb') as handle:
#             self.s_dict = pickle.load(handle)
#
#         self.input_data = np.memmap(self.path + "/model_input.dat", dtype='float32', mode='r', shape=self.size)
#         self.target_data = np.memmap(self.path + "/model_target.dat", dtype='float32', mode='r', shape=self.size)
#
#         prop = [] # list with probabilities
#
#         ss = 0 # sum over all the batches
#         for val in self.s_dict.values():
#             prop.append(val[2])
#             ss += val[2]
#
#         self.prop = np.array(prop) / ss
#         self.length = ss
#
#         self.gen = iter(self.create_data(self.s_dict))
#
#
#
#     def create_data(self, s_dict):
#         while True:
#             ind = np.random.choice(self.size[0], 1, p = self.prop)
#             shp = s_dict[ind[0] + 1][3] # shape of experiment
#
#             cut_point = np.random.randint(low = 200*30, #remove the first 30 secs
#                                 high = shp[1] - 5*200*60, size = 1)
#                                 # choose the place to cut
#
#             chan = np.random.choice(shp[0], 1)
#
#             inp = self.input_data[ind[0], chan[0], cut_point[0]:cut_point[0]+60*5*200]
#             inp = torch.tensor(inp).view(1, 60*5*200)
#             tar = self.target_data[ind[0], chan[0], cut_point[0]:cut_point[0]+60*5*200]
#             tar = torch.tensor(tar).view(1, 60*5*200)
#             # #inp = self.ls[0][0][chan][cut_point[i]:cut_point[i]+60*5*200]
#             # #tar = self.ls[1][0][chan][cut_point[i]:cut_point[i]+60*5*200]
#
#             #tar = torch.cat((tar[0], -1*(tar[0] - 1))).view(2, 60*5*200)
#             yield inp, tar, (ind[0], chan[0], cut_point[0])
#
#
#     def clear_ram(self, index):
#         """
#         This function is for clearing the ram.
#         """
#         if index % 1000 == 0:
#             del self.input_data
#             del self.target_data
#             self.input_data = np.memmap(self.path + "/model_input.dat", dtype='float32', mode='r', shape=self.size)
#             self.target_data = np.memmap(self.path + "/model_target.dat", dtype='float32', mode='r', shape=self.size)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         inp, tar, chan = next(self.gen)
#         inp = inp.to(self.device)
#         tar = tar.to(self.device)
#         self.clear_ram(idx)
#         return inp, tar, chan


class shuffle_5min(Dataset):
    """
    This dataloader loads random 5 minute intervals from a random patient.
    """
    def __init__(self, path: str, series_dict: str, size: tuple, device,
                 seed = None, length = None):
        """
        Args:
            path (str): path to the input & target folder.
            series_dict (list): name of dict for data.
            size : (number of experiments, number of max. channels, longest series)
            device (class 'torch.device'): which pytorch device the data should
            be sent to.
        """

        self.device = device
        self.size = size
        self.path = path
        self.seed = seed

        with open(path + "/" + series_dict, 'rb') as handle:
            self.s_dict = pickle.load(handle)

        self.input_data = np.memmap(self.path + "/model_input.dat", dtype='float32', mode='r', shape=self.size)
        self.target_data = np.memmap(self.path + "/model_target.dat", dtype='float32', mode='r', shape=self.size)

        prop = [] # list with probabilities

        ss = 0 # sum over all the batches
        for val in self.s_dict.values():
            prop.append(val[2])
            ss += val[2]

        self.prop = np.array(prop) / ss

        if length:
            self.length = length
        else:
            self.length = ss


        if not(seed):
            self.rng = np.random.default_rng(self.seed)
            self.gen = iter(self.create_data(self.s_dict, self.rng))




    def create_data(self, s_dict, rng):
        while True:
            ind = rng.choice(self.size[0], 1, p = self.prop)
            shp = s_dict[ind[0] + 1][3] # shape of experiment

            cut_point = rng.integers(low = 200*30, #remove the first 30 secs
                                high = shp[1] - 5*200*60, size = 1)
                                # choose the place to cut

            chan = rng.choice(shp[0], 1)

            inp = self.input_data[ind[0], chan[0], cut_point[0]:cut_point[0]+60*5*200]
            inp = torch.tensor(inp).view(1, 60*5*200)
            tar = self.target_data[ind[0], chan[0], cut_point[0]:cut_point[0]+60*5*200]
            tar = torch.tensor(tar).view(1, 60*5*200)
            # #inp = self.ls[0][0][chan][cut_point[i]:cut_point[i]+60*5*200]
            # #tar = self.ls[1][0][chan][cut_point[i]:cut_point[i]+60*5*200]

            #tar = torch.cat((tar[0], -1*(tar[0] - 1))).view(2, 60*5*200)
            yield inp, tar, (ind[0], chan[0], cut_point[0])


    def clear_ram(self, index):
        """
        This function is for clearing the ram.
        """
        if index % 1000 == 0:
            del self.input_data
            del self.target_data
            self.input_data = np.memmap(self.path + "/model_input.dat", dtype='float32', mode='r', shape=self.size)
            self.target_data = np.memmap(self.path + "/model_target.dat", dtype='float32', mode='r', shape=self.size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.seed:
            if idx == 0:
                self.rng = np.random.default_rng(self.seed)
                self.gen = iter(self.create_data(self.s_dict, self.rng))

        inp, tar, chan = next(self.gen)
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        self.clear_ram(idx)
        return inp, tar, chan



class testload_5min(Dataset):
    """
    This dataloader loads random 5 minute intervals from a random patient.
    """
    def __init__(self, path: str, series_dict: str, size: tuple, device):
        """
        Args:
            path (str): path to the input & target folder.
            series_dict (list): name of dict for data.
            size : (number of experiments, number of max. channels, longest series)
            device (class 'torch.device'): which pytorch device the data should
            be sent to.
        """

        self.device = device
        self.size = size
        self.path = path

        with open(path + "/" + series_dict, 'rb') as handle:
            self.s_dict = pickle.load(handle)

        self.input_data = np.memmap(self.path + "/model_input.dat", dtype='float32', mode='r', shape=self.size)
        self.target_data = np.memmap(self.path + "/model_target.dat", dtype='float32', mode='r', shape=self.size)

        ss = 0 # sum over all the batches
        for val in self.s_dict.values():
            jj = val[3][0]*math.ceil((val[3][1]-20*300) / (5*60*200))
            ss += jj

        self.length = ss

        self.gen = self.load_data(self.s_dict)


    def load_data(self, s_dict, first_drop = 200*30):
        exp_nr = -1
        for ind in s_dict.values():
            exp_nr += 1
            shp = ind[3]
            for chan in range(shp[0]): # number of channels
                for cut_point in range(first_drop, shp[1], 5*60*200): # cut the experiment

                    clear_point = 0 # if the series is not long enough this is used

                    if (w := (shp[1] - cut_point)) < 5*60*200: # There is not enough data!
                        clear_point = cut_point + w
                        inp = torch.zeros(5*60*200)
                        tar = torch.zeros(5*60*200)
                        inp[:w] = torch.Tensor(self.input_data[exp_nr, chan, cut_point:shp[1]]).clone().detach()
                        inp = torch.tensor(inp).view(1, 60*5*200)
                        tar[:w] = torch.Tensor(self.target_data[exp_nr, chan, cut_point:shp[1]]).clone().detach()
                        tar = torch.tensor(tar).view(1, 60*5*200)
                        # if there is not enough data, the point "clear_point"
                        # will be the position on which data is no longer
                        # generated from the experiment. Thus the range:
                        # tar[clear_point:] should not be used for evalutaing
                        # the model, since this is not "true" data
                    else:
                        inp = self.input_data[exp_nr, chan, cut_point:cut_point+60*5*200]
                        inp = torch.tensor(inp).view(1, 60*5*200).clone().detach()
                        tar = self.target_data[exp_nr, chan, cut_point:cut_point+60*5*200]
                        tar = torch.tensor(tar).view(1, 60*5*200).clone().detach()

                    yield inp, tar, (exp_nr, chan, cut_point, clear_point)


    def clear_ram(self, index):
        """
        This function is for clearing the ram.
        """
        if index % 1000 == 0:
            del self.input_data
            del self.target_data
            self.input_data = np.memmap(self.path + "/model_input.dat", dtype='float32', mode='r', shape=self.size)
            self.target_data = np.memmap(self.path + "/model_target.dat", dtype='float32', mode='r', shape=self.size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inp, tar, meta = next(self.gen)
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        self.clear_ram(idx)
        return inp, tar, meta
