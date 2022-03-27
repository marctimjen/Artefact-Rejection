from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import math

class load_whole_data(Dataset): # Dataset
    """
    This dataloader loads the tensor input and target in whole
    """
    def __init__(self, path: str, ind: list):
        """
        Args:
            path (str): path to the input & target folder.
            ind (list): list of indices for which pictures to load.
            device (class 'torch.device'): which pytorch device the data should
            be sent to.
        """

        self.device = "cpu"
        self.imgs_path = path
        self.data = []
        for i in ind:
            self.data.append([self.imgs_path + f"/model_input ({i}).pt",
                        self.imgs_path + f"/model_target ({i}).pt"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_path, target_path = self.data[idx] # path for target + input

        inp = torch.load(input_path) # load the input data
        inp = inp.type(torch.float).to(self.device)

        tar = torch.load(target_path) # load the target data
        tar = tar.type(torch.float).to(self.device)

        return inp, tar




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

        length = math.floor((self.size[1]/(250*60*5)))*self.size[0]
            # the amount of total possible cuts

        self.length = int(min((75 - 75 % self.size[0]), length))
            # make sure that no more than 75 samples is taken from the same
            # individual

        self.gen = iter(self.create_data(self.length))


    def create_data(self, nr_of_cuts):
        cut_point = np.random.randint(low = 250*30, #remove the first 30 secs
                            high = self.size[1] - 5*250*60, size = nr_of_cuts)
                            # choose the place to cut

        cuts_pr_chan = nr_of_cuts/self.ls[0][0].shape[0]
            # the amount of cuts pr channel

        for i in range(nr_of_cuts):
            chan = int(i//cuts_pr_chan) # the given channel
            inp = self.ls[0][0][chan][cut_point[i]:cut_point[i]+60*5*250].view(1, 60*5*250)
            tar = self.ls[1][0][chan][cut_point[i]:cut_point[i]+60*5*250].view(1, 60*5*250)
            #inp = self.ls[0][0][chan][cut_point[i]:cut_point[i]+60*5*250]
            #tar = self.ls[1][0][chan][cut_point[i]:cut_point[i]+60*5*250]


            #tar = torch.cat((tar[0], -1*(tar[0] - 1))).view(2, 60*5*250)
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
        extra = 250*60*5 - (w := (ls[0][0].shape[1]-30*250) % (250*60*5))


        if w: # if w is not equal to 0, then zero-pad is needed:
            # zero pad:
            self.ls[0] = F.pad(self.ls[0], (0, extra), "constant", 0.0)
            self.ls[1] = F.pad(self.ls[1], (0, extra), "constant", 0.0)

            self.size[1] = self.size[1] + extra


        self.length = math.floor((self.size[1]-30*250)/(250*60*5))*self.size[0]
            # the amount of total possible cuts

        self.gen = iter(self.cut_data())



    def cut_data(self):
        for chan in range(self.size[1]):
            for cut_point in range(30*250, self.size[1], 250*5*60):
                inp = self.ls[0][0][chan][cut_point:cut_point+60*5*250].view(1, 60*5*250)
                tar = self.ls[1][0][chan][cut_point:cut_point+60*5*250].view(1, 60*5*250)
                yield (inp, tar, chan, cut_point)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        inp, tar, chan, cut = next(self.gen)
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        return inp, tar, chan, cut
