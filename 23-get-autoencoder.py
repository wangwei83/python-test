import torch
import torch.nn as nn

def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # 编码器部分
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        
        # 解码器部分
        nn.ConvTranspose2d(in_channels=out_channels, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid()
    )

# 创建自动编码器模型
autoencoder = get_autoencoder()
print(autoencoder)
