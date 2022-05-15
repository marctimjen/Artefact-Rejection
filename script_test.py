import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from torch.optim import SGD, Adam
import torchvision.models as models

resnet18 = models.resnet18()

samples = 500
b_size = 10
epochs = 5

end_lr = 1
start_lr = 0.75

# lam = lambda x: math.exp(x * math.log(end_lr / start_lr) \
#                         / (epochs * samples / b_size))


ls = []

optimizer = SGD(resnet18.parameters(), lr=start_lr)

# scheduler = LambdaLR(optimizer, lam)

scheduler = CyclicLR(optimizer, base_lr=start_lr, max_lr=end_lr,
                     step_size_up=(epochs*(samples/b_size)/6),
                     cycle_momentum=False)

for i in range(epochs):
    for j in range(int(samples/b_size)):
        ls.append(optimizer.param_groups[0]['lr'])
        scheduler.step()


print(len(ls))

print((epochs*(samples/b_size)/6))

plt.plot(ls)
plt.show()
