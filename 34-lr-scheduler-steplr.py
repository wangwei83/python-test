import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
model = SimpleModel()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 定义学习率调度器
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# 假设我们有一些数据加载器
train_loader = [torch.randn(32, 10) for _ in range(10)]  # 模拟训练数据

# 训练循环
for epoch in range(10):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, torch.randn(32, 1))  # 模拟目标
        loss.backward()
        optimizer.step()
    
    # 每个epoch结束后更新学习率
    scheduler.step()
    
    # 打印当前学习率
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, Learning Rate: {current_lr:.6f}, Loss: {loss.item():.6f}")
