import torch.nn as nn
import torch
import torch.nn.functional as F


# https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html

class Double_Convolution(nn.Module): # two orange arrows (or two light green arrows)
    """
    This class consistute two orange arrows (or two light green arrows) in the U-net figure (3.6).
    The job of this function is to do a double convolutions.

    Doctesting the results:
    >>> Double_Convolution.forward(Double_Convolution(1, 20), torch.zeros(1, 1, 60)).shape
    torch.Size([1, 20, 56])
    >>> Double_Convolution.forward(Double_Convolution(1, 20, True), torch.zeros(1, 1, 60)).shape
    torch.Size([1, 20, 64])
    >>> Double_Convolution.forward(Double_Convolution(1, 1, True), torch.zeros(1, 1, 6))
    tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]],
           grad_fn=<LeakyReluBackward0>)
    >>> Double_Convolution.forward(Double_Convolution(1, 1), torch.zeros(1, 1, 6))
    tensor([[[0., 0.]]], grad_fn=<LeakyReluBackward0>)
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
        self.relu = nn.LeakyReLU(1/5.5)

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
    This class constitute one red and two orange arrows in the U-net figure.
    So this is the function that does the down-sampling of the net.

    Doctesting the results:
    >>> Down_Scale.forward(Down_Scale(1, 20), torch.zeros(1, 1, 60)).shape
    torch.Size([1, 20, 26])
    >>> Down_Scale.forward(Down_Scale(1, 1), torch.zeros(1, 1, 12))
    tensor([[[0., 0.]]], grad_fn=<LeakyReluBackward0>)
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


class Up_Scale(nn.Module): # yellow arrow + green arrow + double_conv
    """
    This class constitute one yellow, a green and two orange arrows in the U-net
    figure. So this is the function that does the up-sampling of the net.

    Doctesting the results:
    >>> Up_Scale.forward(Up_Scale(4, 2), torch.zeros(1, 4, 110), torch.zeros(1, 2, 221)).shape
    torch.Size([1, 2, 217])
    >>> Up_Scale.forward(Up_Scale(160, 80), torch.zeros(1, 160, 7492), torch.zeros(1, 80, 14993)).shape
    torch.Size([1, 80, 14989])
    >>> Up_Scale.forward(Up_Scale(2, 1), torch.zeros(1, 2, 3), torch.zeros(1, 1, 9)).abs()*0
    tensor([[[0., 0., 0., 0., 0.]]], grad_fn=<MulBackward0>)
    >>> Up_Scale.forward(Up_Scale(40, 20, True), torch.zeros(1, 40, 29990), torch.zeros(1, 20, 59996)).shape
    torch.Size([1, 20, 60000])
    >>> Up_Scale.forward(Up_Scale(4, 2, True), torch.zeros(1, 4, 3), torch.zeros(1, 2, 10)).shape
    torch.Size([1, 2, 14])
    >>> Up_Scale.forward(Up_Scale(2, 1, True), torch.zeros(1, 2, 1), torch.zeros(1, 1, 3)).abs()*0
    tensor([[[0., 0., 0., 0., 0., 0., 0.]]], grad_fn=<MulBackward0>)
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
        diffy = y.size()[2] - x.size()[2]

        x = F.pad(x, [diffy // 2, diffy - diffy // 2])
        # make the dimentions fit

        x = torch.cat([y, x], dim=1)
        x = self.doub(x)
        return x


class OutConv(nn.Module): # magenta arrow
    """
    This class constitute light-blue arrows in the U-net figure. So this is the
    function that does the 1x1 convolution and makes the channels fit to the
    desired output.

    Doctesting the results:
    >>> OutConv.forward(OutConv(20, 1), torch.zeros(1, 20, 60000)).shape
    torch.Size([1, 1, 60000])
    """
    def __init__(self, in_channels, n_classes):
        """
        Args:
            in_channels (int): The amount of channels of the input.
            out_channels (int): The amount of channels the output tensor gets.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, n_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        return self.soft(out)


class Unet(nn.Module):
    """
    This class is the network. So it combines the subparts listed above.

    Doctesting the results:
    >>> Unet.forward(Unet(1, 1), torch.zeros(1, 1, 60000)).shape
    torch.Size([1, 1, 60000])
    """
    def __init__(self, n_channels, n_classes):
        """
        Args:
            n_channels (int): The amount of channels of the input.
            n_classes (int): The amount of channels the output tensor gets.
        """
        super(Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = Double_Convolution(n_channels, 20)
        self.down1 = Down_Scale(20, 40)
        self.down2 = Down_Scale(40, 80)
        self.down3 = Down_Scale(80, 160)
        self.up1 = Up_Scale(160, 80)
        self.up2 = Up_Scale(80, 40)
        self.up3 = Up_Scale(40, 20, up_conv = True)
        self.outc = OutConv(20, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x)
        return output


if __name__ == '__main__':

    import doctest
    doctest.testmod()