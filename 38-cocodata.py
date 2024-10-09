import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义图像和注释文件的路径
root_train = '/mnt/smbmount/1_industry_dataset/COCO2017/train2017'
annFile_train = '/mnt/smbmount/1_industry_dataset/COCO2017/annotations/instances_train2017.json'
root_val = '/mnt/smbmount/1_industry_dataset/COCO2017/val2017'
annFile_val = '/mnt/smbmount/1_industry_dataset/COCO2017/annotations/instances_val2017.json'

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练和验证数据集
train_dataset = dset.CocoDetection(root=root_train, annFile=annFile_train, transform=transform)
val_dataset = dset.CocoDetection(root=root_val, annFile=annFile_val, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)


# import torch
# import torchvision.models.detection as models
# from tqdm import tqdm

# # 加载预训练的 Faster R-CNN 模型
# model = models.fasterrcnn_resnet50_fpn(pretrained=True)
# model.train()

# # 将模型移动到 GPU
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)

# # 定义优化器和学习率调度器
# optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# # 混合精度训练
# scaler = torch.cuda.amp.GradScaler()

# # 训练模型
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

#     for images, targets in progress_bar:
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         optimizer.zero_grad()

#         with torch.cuda.amp.autocast():
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#         scaler.scale(losses).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         epoch_loss += losses.item()
#         progress_bar.set_postfix(loss=losses.item())

#     lr_scheduler.step()
#     print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")


# import matplotlib.pyplot as plt

# # 切换模型到评估模式
# model.eval()

# # 获取一个批次的数据
# images, targets = next(iter(val_loader))
# images = list(image.to(device) for image in images)

# # 进行推理
# with torch.no_grad():
#     predictions = model(images)

# # 可视化结果
# def visualize(image, prediction):
#     plt.imshow(image.permute(1, 2, 0).cpu().numpy())
#     for element in prediction:
#         bbox = element['bbox']
#         plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=2))
#     plt.show()

# # 可视化第一个图像及其预测结果
# visualize(images[0], predictions[0]['boxes'])
