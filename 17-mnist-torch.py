import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import gzip
import numpy as np

# 加载 MNIST 数据集
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(16)
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        data = data.reshape(-1, 28, 28, 1)
        data = data / 255.0
        return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

train_images = load_mnist_images('/home/wangwei83/Desktop/python-test/dataset/mnist/train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('/home/wangwei83/Desktop/python-test/dataset/mnist/train-labels-idx1-ubyte.gz')
test_images = load_mnist_images('/home/wangwei83/Desktop/python-test/dataset/mnist/t10k-images-idx3-ubyte.gz')
test_labels = load_mnist_labels('/home/wangwei83/Desktop/python-test/dataset/mnist/t10k-labels-idx1-ubyte.gz')

# 将数据转换为 PyTorch 张量
train_images = torch.tensor(train_images).permute(0, 3, 1, 2)
train_labels = torch.tensor(train_labels)
test_images = torch.tensor(test_images).permute(0, 3, 1, 2)
test_labels = torch.tensor(test_labels)

# 构建数据加载器
train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 构建模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

print("Done!")

import matplotlib.pyplot as plt
# 可视化样本
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(4):
    img = train_images[i].permute(1, 2, 0).numpy().squeeze()
    ax[i].imshow(img, cmap='gray')
    ax[i].set_title(f'Label: {train_labels[i].item()}')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.show()