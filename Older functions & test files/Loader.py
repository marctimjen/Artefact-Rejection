from torch.utils.data import Dataset, DataLoader
import torch


class data_loader(Dataset):
    """
    This dataloader loads the tensor input and target
    """
    def __init__(self, path: str, ind: list, device):
        """
        Args:
            path (str): path to the input & target folder.
            ind (list): list of indices for which pictures to load.
            device (class 'torch.device'): which pytorch device the data should
            be sent to.
        """

        self.device = device
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
