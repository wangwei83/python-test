from torchvision.datasets import ImageFolder

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        # 获取图像和标签
        original_tuple = super().__getitem__(index)
        # 只返回图像，不返回标签
        return original_tuple[0]

# 使用示例
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 加载数据集
dataset = ImageFolderWithoutTarget(root='./data/train', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 获取一个批次的图像
data_iter = iter(dataloader)
images = data_iter.next()

# 打印图像的形状
print(images.size())

# 代码说明
# 继承 ImageFolder 类：创建一个新的类 ImageFolderWithoutTarget，继承自 ImageFolder。
# 重载 __getitem__ 方法：在 __getitem__ 方法中，只返回图像数据，不返回标签。
# 数据预处理和加载：定义数据预处理操作，加载数据集，并使用 DataLoader 创建数据加载器。
# 获取和打印图像：从数据加载器中获取一个批次的图像，并打印其形状。