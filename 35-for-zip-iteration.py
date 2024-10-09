import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import itertools

# 定义简单的数据集
class ExampleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 定义无限数据加载器函数
def infinite_dataloader(data_loader):
    while True:
        for data in data_loader:
            yield data

# 示例数据
data_st = [torch.randn(10) for _ in range(100)]
data_ae = [torch.randn(10) for _ in range(100)]
data_penalty = [torch.randn(10) for _ in range(100)]

# 创建数据集和数据加载器
dataset_st = ExampleDataset(data_st)
dataset_ae = ExampleDataset(data_ae)
dataset_penalty = ExampleDataset(data_penalty)

train_loader_st = DataLoader(dataset_st, batch_size=32, shuffle=True)
train_loader_ae = DataLoader(dataset_ae, batch_size=32, shuffle=True)
penalty_loader = DataLoader(dataset_penalty, batch_size=32, shuffle=True)

# 创建无限数据加载器
train_loader_infinite_st = infinite_dataloader(train_loader_st)
train_loader_infinite_ae = infinite_dataloader(train_loader_ae)
penalty_loader_infinite = infinite_dataloader(penalty_loader)

# 使用 tqdm 创建进度条
tqdm_obj = tqdm(range(100))  # 假设我们迭代100次

# 训练循环
for iteration, (image_st, image_ae), image_penalty in zip(tqdm_obj, zip(train_loader_infinite_st, train_loader_infinite_ae), penalty_loader_infinite):
    # 这里可以添加训练代码
    print(f"Iteration {iteration}: image_st shape: {image_st.shape}, image_ae shape: {image_ae.shape}, image_penalty shape: {image_penalty.shape}")

    # 假设我们只打印前10次迭代
    if iteration >= 10:
        break
