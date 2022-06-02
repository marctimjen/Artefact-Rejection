import torch.nn as nn
import torch

input_size = 1 # the number of series

hidden_size = 5 # hyper para


D = 2 # bc. bi = True
num_layers = 1 # default

batch_size = 2 # hyper para


proj_size = 2 # This allows us to rechive two values
hout = proj_size # since proj_size > 0


seq_len = 20 # length of the sequence

rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True, \
              proj_size = proj_size) # (input_size, hidden)





input = torch.randn(batch_size, seq_len, input_size) # (batch_size, sequence_length, input_size)
input = torch.randn(batch_size, input_size, seq_len)
print(input)
input = input.view(batch_size, -1, input_size)




# h0 = torch.randn(D*num_layers, batch_size, hout) # (D * num_layers, batch_size, hidden)
# c0 = torch.randn(D*num_layers, batch_size, hidden_size) # (D * num_layers, batch_size, hidden)


output, (hn, cn) = rnn(input)

print(input.shape)

print()

print(output)

print(output.shape) #(batch_size, seq_len, D*H_out)



soft = nn.Softmax(dim=2)

out = soft(output)

print(out)
print(out.shape)

# print(output)
# print(sum(i.nelement() for i in rnn.parameters()))

#
# ss = torch.sum(output, 2)
# minusss = ss - 1
#
# # print(ss)
# # print(minusss)
# #
# print()
# print(torch.stack((ss, minusss), dim = 2).shape)
