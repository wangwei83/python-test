import torch

# 创建一个随机张量
x = torch.rand(5, 3)
print(x)

# 检查是否可以使用 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)  # 直接在 GPU 上创建张量
    x = x.to(device)  # 将张量移动到 GPU
    z = x + y
    print(z)
else:
    print("CUDA 不可用")
