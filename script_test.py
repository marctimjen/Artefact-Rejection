import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from torch.optim import SGD, Adam
import torchvision.models as models

resnet18 = models.resnet18()

samples = 500
b_size = 10
epochs = 5

end_lr = 2.5
start_lr = 0.0001

lam = lambda x: math.exp(x * math.log(end_lr / start_lr) \
                        / (epochs * samples / b_size))


ls = []

optimizer = SGD(resnet18.parameters(), lr=start_lr)

scheduler = LambdaLR(optimizer, lam)

for i in range(epochs):
    for j in range(int(samples/b_size)):
        ls.append(optimizer.param_groups[0]['lr'])
        scheduler.step()


print(len(ls))

plt.plot(ls)
plt.show()
