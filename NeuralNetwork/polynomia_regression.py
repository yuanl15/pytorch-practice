import torch
from torch.autograd import Variable
from torch import nn
from torch import optim

# 将输入转换为矩阵
def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])


def f(x):
    return x.mm(w_target) + b_target[0]


# 随机生成训练数据
def get_batch(batch_size=32):
    """Builds a batch i.e. (x, f(x)) pair."""
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)


# define model
class PolyRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.poly = nn.Linear(3, 1)

    def forward(self, x):
        out = self.poly(x)
        return out


if torch.cuda.is_available():
    model = PolyRegression().cuda()
else:
    model = PolyRegression()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    # 获取训练数据
    x_train, y_train = get_batch()

    # 模型输出
    output = model(x_train)

    # 计算损失
    loss = criterion(output, y_train)

    print(loss.item())

    # 优化器归零
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 优化起更新参数
    optimizer.step()

    epoch += 1
    if loss < 1e-3:
        break
