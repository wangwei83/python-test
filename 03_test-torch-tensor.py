# x = torch.rand(5, 3)
# tensor([[0.7270, 0.1998, 0.3044],
#         [0.9977, 0.5624, 0.1866],
#         [0.0482, 0.0637, 0.8616],
#         [0.4901, 0.5053, 0.4819],
#         [0.7369, 0.7443, 0.4554]])
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], device='cuda:0')
# tensor([[1.7270, 1.1998, 1.3044],
#         [1.9977, 1.5624, 1.1866],
#         [1.0482, 1.0637, 1.8616],
#         [1.4901, 1.5053, 1.4819],
#         [1.7369, 1.7443, 1.4554]], device='cuda:0')
import torch

# 创建一个随机张量
x = torch.rand(5, 3)
print(x)

# 检查是否可以使用 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)  # 直接在 GPU 上创建张量
    print(y)
    x = x.to(device)  # 将张量移动到 GPU
    z = x + y
    print(z)
else:
    print("CUDA 不可用")
