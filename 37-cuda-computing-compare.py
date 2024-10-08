import torch
import time

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# 创建一个大的随机张量
tensor_size = (10000, 10000)
tensor_cpu = torch.randn(tensor_size)

# 将张量移动到GPU
tensor_gpu = tensor_cpu.cuda() if torch.cuda.is_available() else tensor_cpu



# 在GPU上进行矩阵乘法并计时
start_time = time.time()
result_gpu = torch.matmul(tensor_gpu, tensor_gpu)
gpu_time = time.time() - start_time

# 在CPU上进行矩阵乘法并计时
start_time = time.time()
result_cpu = torch.matmul(tensor_cpu, tensor_cpu)
cpu_time = time.time() - start_time

print(f"CPU计算时间: {cpu_time:.6f} 秒")
print(f"GPU计算时间: {gpu_time:.6f} 秒")


# (efficientad-env) wangwei83@wangwei83-System-Product-Name:~/Desktop/python-test$ /home/wangwei83/miniconda3/envs/efficientad-env/bin/python /home/wangwei83/Desktop/python-test/37-cuda-computing-compare.py
# GPU is available
# CPU计算时间: 4.944673 秒
# GPU计算时间: 0.833279 秒