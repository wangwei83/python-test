import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNet()

# 将模型移动到GPU
if torch.cuda.is_available():
    model.cuda()
    print("Model moved to GPU")

# 创建一些示例数据并移动到GPU
input_data = torch.randn(5, 10)
target = torch.randn(5, 1)

if torch.cuda.is_available():
    input_data = input_data.cuda()
    target = target.cuda()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 前向传播
output = model(input_data)
loss = criterion(output, target)

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())
