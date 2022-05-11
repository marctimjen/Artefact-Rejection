import torch


v = torch.rand((3, 6, 2))

# Skal laves til (10, 2, 60000)

# pred = y_pred.transpose(1, 2).reshape(-1, 2).type(fl)

ef = v.transpose(1, 2)

print(v)

print(ef)
print(ef.shape)
