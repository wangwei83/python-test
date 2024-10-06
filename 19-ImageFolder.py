# from torchvision.datasets import ImageFolder
# from torchvision import transforms

# # Define your transformations
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Create the dataset
# dataset = ImageFolder(root='/home/wangwei83/Desktop/python-test/dataset/cifar/data', transform=transform)

# # Example: Accessing the first image and its label
# img, label = dataset[0]

# print(f'Image shape: {img.shape}, Label: {label}')



import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建数据集
dataset = ImageFolder(root='/home/wangwei83/Desktop/python-test/dataset/cifar/data', transform=transform)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 访问数据集中的第一个样本
img, label = dataset[12]
print(f'Image shape: {img.shape}, Label: {label}')
