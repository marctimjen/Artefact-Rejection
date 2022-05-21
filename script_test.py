import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from torch.optim import SGD, Adam
import torchvision.models as models

resnet18 = models.resnet18()

samples = 11141 - 1
b_size = 10
epochs = 100

end_lr = 0.268
start_lr = 0.216

# lam = lambda x: math.exp(x * math.log(end_lr / start_lr) \
#                         / (epochs * samples / b_size))


ls = []

optimizer = Adam(resnet18.parameters(), lr=start_lr)

try:
    mom = optimizer.param_groups[0]['momentum']
    mom = True
except:
    mom = False

print(mom)

# scheduler = LambdaLR(optimizer, lam)

scheduler = CyclicLR(optimizer, base_lr=start_lr, max_lr=end_lr,
                     step_size_up=(samples/b_size)*5,
                     cycle_momentum=False)

for i in range(epochs):
    for j in range(int(samples/b_size)):
        ls.append(optimizer.param_groups[0]['lr'])
        scheduler.step()


# print(len(ls))
#
# print((epochs*(samples/b_size)/6))

plt.plot(ls)
plt.show()
