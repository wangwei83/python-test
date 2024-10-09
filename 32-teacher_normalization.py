import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 定义一个简单的数据集
class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 计算模型输出的均值和标准差
def teacher_normalization(model, data_loader):
    model.eval()
    outputs = []
    with torch.no_grad():
        for data in data_loader:
            output = model(data)
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    mean = outputs.mean(dim=0)
    std = outputs.std(dim=0)
    return mean, std

# 示例数据
data = [torch.randn(10) for _ in range(100)]
print(data)
dataset = ExampleDataset(data)
print(dataset)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(train_loader)

# 创建模型实例
teacher = SimpleModel()
print(teacher)

# 计算教师模型的均值和标准差
teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

print("Teacher Mean:", teacher_mean)
print("Teacher Std:", teacher_std)
