
import torch
from torch.utils.data import DataLoader, TensorDataset

# 创建一些示例数据
data = torch.randn(50, 3)
labels = torch.randint(0, 2, (50,))
print('example data and labels')
print(data, labels)


# 创建一个 TensorDataset
dataset = TensorDataset(data, labels)

# 创建一个 DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# # 迭代 DataLoader
# for batch_data, batch_labels in dataloader:
#     print('batch data and labels')
#     print(batch_data, batch_labels)


# 迭代 DataLoader 并区分不同的批次编号
for batch_num, (batch_data, batch_labels) in enumerate(dataloader, 1):
    print(f'Batch {batch_num}:')
    # print('Batch data and labels:')
    print(batch_data, batch_labels)