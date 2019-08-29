import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
import numpy as np

############# 使用逻辑回归对点进行二分类 ################

with open('logistic-data.txt') as f:
    data_list = f.readlines()
    data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

x0 = list(filter(lambda x: x[-1] == 0.0, data))
x1 = list(filter(lambda x: x[-1] == 1.0, data))
plot_x0_0 = [i[0] for i in x0]
plot_x0_1 = [i[1] for i in x0]
plot_x1_0 = [i[0] for i in x1]
plot_x1_1 = [i[1] for i in x1]

plt.plot(plot_x0_0, plot_x0_1, 'ro', label='x_0')
plt.plot(plot_x1_0, plot_x1_1, 'bo', label='x1')
plt.legend(loc='best')
x_data = torch.FloatTensor([[i[0], i[1]] for i in data])
y_data = torch.FloatTensor([i[-1] for i in data]).unsqueeze(1)


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = nn.Linear(2, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x

model = LogisticRegression()
if torch.cuda.is_available():
    model.cuda()

# 定义BCE二分类损失函数
criterion = nn.BCELoss()
# 优化器选择带冲量的SGD
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(50000):
    if torch.cuda.is_available():
        x = Variable(x_data).cuda()
        y = Variable(y_data).cuda()
    else:
        x = Variable(x_data)
        y = Variable(y_data)

    # 前向推导
    out = model(x_data)
    loss = criterion(out, y)
    print(loss.item())
    mask = out.ge(0.5).float()
    correct = (mask == y).sum()
    acc = correct.data.item() / x.size(0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print('*'*10)
        print('epoch {}'.format(epoch+1))
        print('loss is {:.4f}'.format(loss))
        print('acc is {:.4f}'.format(acc))