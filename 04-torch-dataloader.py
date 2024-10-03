import torch
from torch.utils.data import DataLoader, TensorDataset

# 创建一些示例数据
data = torch.randn(100, 3)
labels = torch.randint(0, 2, (100,))

# 创建一个 TensorDataset
dataset = TensorDataset(data, labels)

# 创建一个 DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 迭代 DataLoader
for batch_data, batch_labels in dataloader:
    print(batch_data, batch_labels)
