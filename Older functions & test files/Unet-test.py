import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
import numpy as np
import time


# create network:

class Double_Convolution(nn.Module): # Blue arrow
    """
    This class constitute two dark blue arrows in the U-net figure. So it does
    a double convolution.
    """
    def __init__(self, in_channels, out_channels, up_conv = False):
        """
        Args:
            in_channels (int): The amount of channels of the input.
            out_channels (int): The amount of channels the output tensor gets.
            up_conv (bool): Indicator if the function should do an
            up-convolution.
        """
        super().__init__()
        if up_conv:
            self.conv1 = nn.ConvTranspose1d(in_channels, out_channels,
                                            kernel_size=3)

            self.conv2 = nn.ConvTranspose1d(out_channels, out_channels,
                                            kernel_size=3)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3)

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


class Down_Scale(nn.Module): # red arrow + double_conv
    """
    This class constitute one red and two dark blue arrows in the U-net figure.
    So this is the function that does the down-sampling of the net.
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): The amount of channels of the input.
            out_channels (int): The amount of channels the output tensor gets.
        """
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.doub = Double_Convolution(in_channels, out_channels)
    def forward(self, x):
        x = self.pool(x)
        x = self.doub(x)
        return x


class Up_Scale(nn.Module): # green arrow + double_conv
    """
    This class constitute one green and two dark blue arrows in the U-net
    figure. So this is the function that does the up-sampling of the net.
    """
    def __init__(self, in_channels, out_channels, up_conv = False):
        """
        Args:
            in_channels (int): The amount of channels of the input.
            out_channels (int): The amount of channels the output tensor gets.
            up_conv (bool): Indicator if the function should do an
            up-convolution.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = nn.ConvTranspose1d(in_channels, out_channels,
                                            kernel_size = 2)
        self.doub = Double_Convolution(in_channels, out_channels, up_conv)

    def forward(self, x, y):
        x = self.up(x)
        x = self.up_conv1(x)

        diffY = y.size()[2] - x.size()[2]
        #diffX = y.size()[3] - x.size()[3], diffX // 2, diffX - diffX // 2

        x = F.pad(x, [diffY // 2, diffY - diffY // 2]) # make the dimentions fit

        x = torch.cat([y, x], dim=1)
        x = self.doub(x)
        return x

class OutConv(nn.Module): # light-blue arrow
    """
    This class constitute light-blue arrows in the U-net figure. So this is the
    function that does the 1x1 convolution and makes the channels fit to the
    desired output.
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): The amount of channels of the input.
            out_channels (int): The amount of channels the output tensor gets.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        return self.soft(x)


class GTA_Unet(nn.Module):
    """
    This class is the network. So it combines the subparts listed above.
    """
    def __init__(self, n_channels, n_classes):
        """
        Args:
            n_channels (int): The amount of channels of the input.
            n_classes (int): The amount of channels the output tensor gets.
        """
        super(GTA_Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = Double_Convolution(n_channels, 64)
        self.down1 = Down_Scale(64, 128)
        self.down2 = Down_Scale(128, 256)
        self.down3 = Down_Scale(256, 512)
        self.down4 = Down_Scale(512, 1024)
        self.up1 = Up_Scale(1024, 512)
        self.up2 = Up_Scale(512, 256)
        self.up3 = Up_Scale(256, 128)
        self.up4 = Up_Scale(128, 64, up_conv = True)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output




input_path = "C:/Users/Marc/Desktop/model_data/model_input (309).pt"
target_path = "C:/Users/Marc/Desktop/model_data/model_target (309).pt"

inp = torch.load(input_path) # load the input data
tar = torch.load(target_path) # load the target data

ind1 = inp[0][60*5*250:60*10*250].reshape(1, 1, 75000).type(torch.float)
target1 = tar[0][60*5*250:60*10*250].reshape(1, 1, 75000).type(torch.float)

ind2 = inp[0][60*10*250:60*15*250].reshape(1, 1, 75000).type(torch.float)
target2 = tar[0][60*10*250:60*15*250].reshape(1, 1, 75000).type(torch.float)

ind3 = inp[0][60*15*250:60*20*250].reshape(1, 1, 75000).type(torch.float)
target3 = tar[0][60*15*250:60*20*250].reshape(1, 1, 75000).type(torch.float)

input_list = [(ind1, target1), (ind2, target2), (ind3, target3)]

print(ind1.shape)

m = nn.Conv1d(1, 3, 3, stride=1)
output = m(ind1)
print(output.shape)

b = nn.BatchNorm1d(3)
output = b(output)
print(output.shape)

pool = nn.MaxPool1d(2)
output = pool(output)
print(output.shape)

up = nn.Upsample(scale_factor=2, mode='nearest')
output = up(output)
print(output.shape)

lossFunc = nn.MSELoss()
model = GTA_Unet(n_channels = 1, n_classes = 2)

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loss = []

for ind, tar in input_list:

    y_pred = model(ind)
    print(y_pred)
    model.zero_grad()
    loss = lossFunc(y_pred, tar)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())

print(train_loss)
