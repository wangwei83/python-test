# (base) wangwei83@wangwei83-System-Product-Name:~/Desktop/python-test$ conda activate zgp_efficientpy38t113
# (zgp_efficientpy38t113) wangwei83@wangwei83-System-Product-Name:~/Desktop/python-test$ /home/wangwei83/miniconda3/envs/zgp_efficientpy38t113/bin/python /home/wangwei83/Desktop/python-test/04-torch-dataloader.py

# A module that was compiled using NumPy 1.x cannot be run in
# NumPy 2.1.1 as it may crash. To support both 1.x and 2.x
# versions of NumPy, modules must be compiled with NumPy 2.0.
# Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

# If you are a user of the module, the easiest solution will be to
# downgrade to 'numpy<2' or try to upgrade the affected module.
# We expect that some modules will need time to support NumPy 2.

# Traceback (most recent call last):  File "/home/wangwei83/Desktop/python-test/04-torch-dataloader.py", line 6, in <module>
#     data = torch.randn(50, 3)
# /home/wangwei83/Desktop/python-test/04-torch-dataloader.py:6: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /opt/conda/conda-bld/pytorch_1670525552843/work/torch/csrc/utils/tensor_numpy.cpp:77.)
#   data = torch.randn(50, 3)
# example data and labels
# tensor([[ 0.8832,  0.8727,  1.5607],
#         [-0.2336, -0.2363,  0.9795],
#         [ 0.5174,  0.0936,  0.1967],
#         [ 0.2937,  0.0390,  1.3673],
#         [-0.0790,  0.0236, -0.7219],
#         [ 0.0441,  0.9445,  1.1603],
#         [-0.6627,  0.4072,  0.8607],
#         [-0.8737, -0.1559, -1.1588],
#         [-1.1384, -0.3093,  0.0934],
#         [ 1.6196, -0.4915, -0.4205],
#         [ 1.3551,  1.5810, -0.5245],
#         [ 0.0872,  0.3001, -1.3691],
#         [ 1.9108, -0.1126, -0.2945],
#         [ 0.9329, -0.3709,  0.4369],
#         [-1.2609,  0.8302, -1.8031],
#         [ 1.6327, -0.1009, -0.0666],
#         [ 0.1356, -1.1509, -1.6297],
#         [ 0.6184,  0.7177, -1.0190],
#         [-0.7541,  1.5020, -1.0045],
#         [ 1.0551,  0.8592,  0.9411],
#         [ 0.5814, -0.3839,  0.1647],
#         [-1.1917, -1.3294,  1.1448],
#         [-1.5714, -0.4512,  1.5351],
#         [-0.6165,  0.5370, -0.6648],
#         [ 1.0222,  0.5989,  0.9017],
#         [ 0.3362, -1.2162,  0.1970],
#         [ 0.1462,  0.4376, -1.2670],
#         [-0.0777,  0.2477,  0.2931],
#         [-0.6015, -0.4933,  0.8636],
#         [ 0.5165, -1.3437,  2.1992],
#         [ 1.1924, -0.5012, -1.5612],
#         [ 0.3196,  0.1808, -1.3073],
#         [-0.0683, -0.0409, -0.9365],
#         [-0.6016, -0.6105,  1.4400],
#         [-0.5347, -0.6872, -0.7969],
#         [ 1.0950,  0.0736, -0.8634],
#         [-0.8088, -0.3577,  0.9669],
#         [ 0.7466, -0.5301,  0.6240],
#         [ 1.2997,  1.0840, -0.2238],
#         [ 1.0243, -0.7417,  1.1574],
#         [ 0.0061,  0.0173,  0.3615],
#         [ 0.7235,  0.3397,  0.0780],
#         [ 0.3793, -0.7828,  1.1578],
#         [ 0.7430, -0.1131, -0.2256],
#         [-0.9719,  1.0712, -0.3530],
#         [ 1.2656, -0.3716,  0.0801],
#         [-1.0907, -0.2632,  0.9515],
#         [ 0.1253, -0.5425,  2.6260],
#         [ 0.0671,  0.9053,  0.3849],
#         [ 0.6482, -0.1500, -1.2157]]) tensor([1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
#         1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
#         1, 0])
# Batch 1:
# tensor([[ 0.3196,  0.1808, -1.3073],
#         [ 0.6482, -0.1500, -1.2157],
#         [ 1.0243, -0.7417,  1.1574],
#         [ 1.9108, -0.1126, -0.2945],
#         [ 0.3362, -1.2162,  0.1970],
#         [ 0.5174,  0.0936,  0.1967],
#         [ 0.5165, -1.3437,  2.1992],
#         [ 0.7430, -0.1131, -0.2256],
#         [ 0.1356, -1.1509, -1.6297],
#         [-1.0907, -0.2632,  0.9515]]) tensor([1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
# Batch 2:
# tensor([[-1.2609,  0.8302, -1.8031],
#         [ 1.2656, -0.3716,  0.0801],
#         [ 0.0671,  0.9053,  0.3849],
#         [-0.2336, -0.2363,  0.9795],
#         [ 1.1924, -0.5012, -1.5612],
#         [-0.9719,  1.0712, -0.3530],
#         [ 0.1462,  0.4376, -1.2670],
#         [ 0.3793, -0.7828,  1.1578],
#         [ 1.6196, -0.4915, -0.4205],
#         [ 0.7466, -0.5301,  0.6240]]) tensor([0, 0, 1, 0, 0, 0, 1, 1, 0, 0])
# Batch 3:
# tensor([[ 1.0551,  0.8592,  0.9411],
#         [ 0.0441,  0.9445,  1.1603],
#         [ 0.7235,  0.3397,  0.0780],
#         [-0.5347, -0.6872, -0.7969],
#         [ 1.6327, -0.1009, -0.0666],
#         [-0.6165,  0.5370, -0.6648],
#         [-0.8737, -0.1559, -1.1588],
#         [ 1.2997,  1.0840, -0.2238],
#         [-0.0683, -0.0409, -0.9365],
#         [ 1.3551,  1.5810, -0.5245]]) tensor([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
# Batch 4:
# tensor([[-1.5714, -0.4512,  1.5351],
#         [-1.1917, -1.3294,  1.1448],
#         [-0.6627,  0.4072,  0.8607],
#         [-0.6015, -0.4933,  0.8636],
#         [ 0.5814, -0.3839,  0.1647],
#         [-1.1384, -0.3093,  0.0934],
#         [-0.8088, -0.3577,  0.9669],
#         [ 0.9329, -0.3709,  0.4369],
#         [-0.0777,  0.2477,  0.2931],
#         [-0.7541,  1.5020, -1.0045]]) tensor([0, 1, 0, 1, 0, 1, 1, 1, 0, 0])
# Batch 5:
# tensor([[ 1.0950,  0.0736, -0.8634],
#         [ 0.6184,  0.7177, -1.0190],
#         [ 0.0061,  0.0173,  0.3615],
#         [ 0.2937,  0.0390,  1.3673],
#         [-0.0790,  0.0236, -0.7219],
#         [ 0.0872,  0.3001, -1.3691],
#         [ 1.0222,  0.5989,  0.9017],
#         [ 0.1253, -0.5425,  2.6260],
#         [ 0.8832,  0.8727,  1.5607],
#         [-0.6016, -0.6105,  1.4400]]) tensor([1, 1, 0, 1, 1, 0, 1, 0, 1, 0])
# (zgp_efficientpy38t113) wangwei83@wangwei83-System-Product-Name:~/Desktop/python-test$ 
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