import numpy as np
import torch
from torch import nn

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 转换为tensor张量
x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)

class LinearRegression(nn.Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()