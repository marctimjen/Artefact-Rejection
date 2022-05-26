import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTM_net(nn.Module):
    """
    This is a network with the LSTM nodes
    """
    def __init__(self, batch_size, device):
        """
        Args:
            out_channels (int): The amount of channels the output tensor gets.
        """
        super(LSTM_net, self).__init__()


        self.batch_size = batch_size
        self.input_size = 1 # the number of series
        hidden_size = 5 # hyper para

        D = 2 # bc. bi = True
        num_layers = 50 # default (how many cells)


        proj_size = 1 # This allows us to rechive two values
        hout = proj_size # since proj_size > 0

        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, \
                            batch_first=True, bidirectional=True, \
                            proj_size = proj_size)
                            # (input_size, hidden)

        #self.h = torch.zeros(D*num_layers, batch_size, hout).to(device)
        # (D * num_layers, batch_size, hidden)

        #self.c = torch.zeros(D*num_layers, batch_size, hidden_size).to(device)
        # (D * num_layers, batch_size, hidden)


        # implementer LSTM eller GRU HER!! <- før pseudo!!
        # Bidirectional lag - så den kører begge veje.
        # LSTM bestemme outpu dimmentionen - ellers brug en conv eller fully connected.
        # Måske maxpool er fint nok. Kogt de fire outputs ned i en.

        # Kig på batch_first - den kørrer anden konvention, bidirectional = True

        #self.soft = F.log_softmax(dim=1) # Using sigmoid instead of softmax
        self.soft = nn.Softmax(dim=1) # Using sigmoid instead of softmax
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        ind = x.view(self.batch_size, -1, self.input_size)
        out, _ = self.lstm(ind)

        ss = torch.sum(out, 2)
        minusss = 1 - ss

        out = torch.stack((ss, minusss), dim = 1)

        return self.soft(out)
