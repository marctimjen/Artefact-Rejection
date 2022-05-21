import torch.nn as nn
import torch

input_size = 2 # the number of series

hidden_size = 5 # hyper para


D = 2 # bc. bi = True
num_layers = 1 # default

batch_size = 10 # hyper para


proj_size = 1 # This allows us to rechive two values
hout = proj_size # since proj_size > 0


seq_len = 20 # length of the sequence

rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True, \
              proj_size = proj_size) # (input_size, hidden)





input = torch.randn(batch_size - 3, seq_len, input_size) # (batch_size, sequence_length, input_size)

print(input.shape)

h0 = torch.randn(D*num_layers, batch_size, hout) # (D * num_layers, batch_size, hidden)
c0 = torch.randn(D*num_layers, batch_size, hidden_size) # (D * num_layers, batch_size, hidden)


output, (hn, cn) = rnn(input, (h0, c0))

# print(input)
#
# print()
#
# print(output)
#
# print()

ss = torch.sum(output, 2)
minusss = ss - 1

# print(ss)
# print(minusss)
#
# print()
# print(torch.stack((ss, minusss), dim = 2))
