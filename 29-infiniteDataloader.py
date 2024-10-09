import torch
from torch.utils.data import DataLoader, Dataset

# 示例数据集
class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 无限数据加载器函数
def InfiniteDataLoader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)

# 示例数据
data = list(range(100))
dataset = ExampleDataset(data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

# 使用无限数据加载器
train_loader_infinite = InfiniteDataLoader(train_loader)

# 测试无限数据加载器
for i, batch in enumerate(train_loader_infinite):
    print(batch)
    if i > 10:  # 仅打印前10个批次
        break
