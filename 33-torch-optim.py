import torch
import torch.nn as nn
import torch.optim as optim
import itertools

# 定义简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
student = SimpleModel()
autoencoder = SimpleModel()

# 定义优化器
optimizer = optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=1e-4, weight_decay=1e-5)

# 假设我们有一些数据加载器
train_loader = [torch.randn(32, 10) for _ in range(100)]  # 模拟训练数据

# 训练阶段
student.train()
autoencoder.train()
for data in train_loader:
    optimizer.zero_grad()
    student_output = student(data)
    autoencoder_output = autoencoder(data)
    
    # 假设我们有一些损失函数
    student_loss = nn.MSELoss()(student_output, torch.randn(32, 1))  # 模拟目标
    autoencoder_loss = nn.MSELoss()(autoencoder_output, torch.randn(32, 1))  # 模拟目标
    
    # 总损失
    total_loss = student_loss + autoencoder_loss
    total_loss.backward()
    optimizer.step()

    print(f"Student Loss: {student_loss.item()}, Autoencoder Loss: {autoencoder_loss.item()}")
