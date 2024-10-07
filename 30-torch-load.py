import torch
import torch.nn as nn

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
teacher = SimpleModel()

# 假设我们有一个配置对象
class Config:
    weights = 'model_weights.pth'

config = Config()

# 保存模型权重
torch.save(teacher.state_dict(), config.weights)

# 加载模型权重
state_dict = torch.load(config.weights, map_location='cpu')
teacher.load_state_dict(state_dict)

# 验证模型是否正确加载
input_tensor = torch.randn(1, 10)
output = teacher(input_tensor)
print(output)
