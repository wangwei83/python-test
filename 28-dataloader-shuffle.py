
# 以下是一个使用 DataLoader 的示例，其中包含 train_set 数据集，批次大小为 1，数据打乱，使用 4 个工作进程，并启用锁页内存：
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 加载数据集
train_set = MNIST(root='./data', train=True, download=True, transform=ToTensor())

# 创建 DataLoader
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

# 打印一些数据
for i, (images, labels) in enumerate(train_loader):
    print(f"Batch {i+1}:")
    print(f"Images: {images}")
    print(f"Labels: {labels}")
    if i == 2:  # 仅打印前三个批次
        break
