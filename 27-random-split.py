import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 加载数据集
full_train_set = MNIST(root='./data', train=True, download=True, transform=ToTensor())

# 定义训练集和验证集的大小
train_size = int(0.8 * len(full_train_set))
validation_size = len(full_train_set) - train_size

# 设置随机数生成器种子
seed = 42
rng = torch.Generator().manual_seed(seed)

# 划分数据集
train_dataset, validation_dataset = random_split(full_train_set, [train_size, validation_size], generator=rng)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# 打印划分结果
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(validation_dataset)}")

# rm -rf data
