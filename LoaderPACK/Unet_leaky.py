import torch.nn as nn
import torch
import torch.nn.functional as F


# https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html

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

class OutConv_lstm(nn.Module): # light-blue arrow
    """
    This class constitute light-blue arrows in the U-net figure. So this is the
    function that does the 1x1 convolution and makes the channels fit to the
    desired output.
    """
    def __init__(self, in_channels, batch_size, device):
        """
        Args:
            in_channels (int): The amount of channels of the input.
            out_channels (int): The amount of channels the output tensor gets.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)

        input_size = 2 # the number of series
        hidden_size = 5 # hyper para

        D = 2 # bc. bi = True
        num_layers = 1 # default


        proj_size = 1 # This allows us to rechive two values
        hout = proj_size # since proj_size > 0


        seq_len = 200*5*60 # length of the sequence


        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, \
                            bidirectional=True, proj_size = proj_size)
                            # (input_size, hidden)

        self.h = torch.randn(D*num_layers, batch_size, hout).to(device)
        # (D * num_layers, batch_size, hidden)

        self.c = torch.randn(D*num_layers, batch_size, hidden_size).to(device)
        # (D * num_layers, batch_size, hidden)


        # implementer LSTM eller GRU HER!! <- før pseudo!!
        # Bidirectional lag - så den kører begge veje.
        # LSTM bestemme outpu dimmentionen - ellers brug en conv eller fully connected.
        # Måske maxpool er fint nok. Kogt de fire outputs ned i en.

        # Kig på batch_first - den kørrer anden konvention, bidirectional = True

        self.soft = nn.Softmax(dim=1) # Using sigmoid instead of softmax
        #self.sig = nn.Sigmoid()

    def forward(self, x, inp):
        x = self.conv(x)
        stack_att = torch.stack((x, inp), dim = 3)
        stack_att = torch.squeeze(stack_att, 1)
        out, _ = self.lstm(stack_att, (self.h, self.c))

        ss = torch.sum(out, 2)
        minusss = ss - 1

        out = torch.stack((ss, minusss), dim = 2)

        return self.soft(out)


class OutConv(nn.Module): # light-blue arrow
    """
    This class constitute light-blue arrows in the U-net figure. So this is the
    function that does the 1x1 convolution and makes the channels fit to the
    desired output.
    """
    def __init__(self, in_channels, n_classes):
        """
        Args:
            in_channels (int): The amount of channels of the input.
            out_channels (int): The amount of channels the output tensor gets.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, n_classes, kernel_size=1)


        # implementer LSTM eller GRU HER!! <- før pseudo!!
        # Bidirectional lag - så den kører begge veje.
        # LSTM bestemme outpu dimmentionen - ellers brug en conv eller fully connected.
        # Måske maxpool er fint nok. Kogt de fire outputs ned i en.

        # Kig på batch_first - den kørrer anden konvention, bidirectional = True

        self.soft = nn.Softmax(dim=1) # Using sigmoid instead of softmax
        #self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        return self.soft(out)

#class Unet(nn.Module):
#    """
#    This class is the network. So it combines the subparts listed above.
#    """
#    def __init__(self, n_channels, n_classes):
#        """
#        Args:
#            n_channels (int): The amount of channels of the input.
#            n_classes (int): The amount of channels the output tensor gets.
#        """
#        super(Unet, self).__init__()
#
#        self.n_channels = n_channels
#        self.n_classes = n_classes
#
#        self.inc = Double_Convolution(n_channels, 64)
#        self.down1 = Down_Scale(64, 128)
#        self.down2 = Down_Scale(128, 256)
#        self.down3 = Down_Scale(256, 512)
#        self.down4 = Down_Scale(512, 1024)
#        self.up1 = Up_Scale(1024, 512)
#        self.up2 = Up_Scale(512, 256)
#        self.up3 = Up_Scale(256, 128)
#        self.up4 = Up_Scale(128, 64, up_conv = True)
#        self.outc = OutConv(64, n_classes)
#
#    def forward(self, x):
#        x1 = self.inc(x)
#        x2 = self.down1(x1)
#        x3 = self.down2(x2)
#        x4 = self.down3(x3)
#        x5 = self.down4(x4)
#        x = self.up1(x5, x4)
#        x = self.up2(x, x3)
#        x = self.up3(x, x2)
#        x = self.up4(x, x1)
#        output = self.outc(x)
#        return output


# class Unet_leaky(nn.Module):
#     """
#     This class is the network. So it combines the subparts listed above.
#     """
#     def __init__(self, n_channels, n_classes):
#         """
#         Args:
#             n_channels (int): The amount of channels of the input.
#             n_classes (int): The amount of channels the output tensor gets.
#         """
#         super(Unet_leaky, self).__init__()
#
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#
#         self.inc = Double_Convolution(n_channels, 10)
#         self.down1 = Down_Scale(10, 20)
#         self.down2 = Down_Scale(20, 40)
#         self.down3 = Down_Scale(40, 80)
#         self.up1 = Up_Scale(80, 40)
#         self.up2 = Up_Scale(40, 20)
#         self.up3 = Up_Scale(20, 10, up_conv = True)
#         self.outc = OutConv(10, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x = self.up1(x4, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)
#         output = self.outc(x)
#         return output


class Unet_leaky_lstm(nn.Module):
    """
    This class is the network. So it combines the subparts listed above.
    """
    def __init__(self, n_channels, batch_size, device):
        """
        Args:
            n_channels (int): The amount of channels of the input.
            n_classes (int): The amount of channels the output tensor gets.
        """
        super(Unet_leaky_lstm, self).__init__()

        self.inc = Double_Convolution(n_channels, 20)
        self.down1 = Down_Scale(20, 40)
        self.down2 = Down_Scale(40, 80)
        self.down3 = Down_Scale(80, 160)
        self.up1 = Up_Scale(160, 80)
        self.up2 = Up_Scale(80, 40)
        self.up3 = Up_Scale(40, 20, up_conv = True)
        self.outc = OutConv_lstm(20, batch_size, device)

    def forward(self, inp):
        x1 = self.inc(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.outc(x, inp)
        return output


class Unet_leaky(nn.Module):
    """
    This class is the network. So it combines the subparts listed above.
    """
    def __init__(self, n_channels, n_classes):
        """
        Args:
            n_channels (int): The amount of channels of the input.
            n_classes (int): The amount of channels the output tensor gets.
        """
        super(Unet_leaky, self).__init__()

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
