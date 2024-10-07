import torch

# 设置随机数种子
seed = 42
rng = torch.Generator().manual_seed(seed)

# 使用生成器创建随机张量
random_tensor = torch.rand(3, 3, generator=rng)
print(random_tensor)
