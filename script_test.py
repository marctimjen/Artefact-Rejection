import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from torch.optim import SGD, Adam
import torchvision.models as models

resnet18 = models.resnet18()

samples = 400
b_size = 10
epochs = 10

end_lr = 10
start_lr = 1

lam = lambda x: math.exp(x * math.log(end_lr / start_lr) \
                        / (epochs * samples / b_size))


ls = []

optimizer = SGD(resnet18.parameters(), lr=start_lr)

scheduler = LambdaLR(optimizer, lam)

for i in range(epochs):
    for j in range(int(samples/b_size)):
        ls.append(optimizer.param_groups[0]['lr'])
        scheduler.step()


plt.plot(ls)
plt.show()
