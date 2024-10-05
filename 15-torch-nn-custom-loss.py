import torch
from torch import nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean((output - target) ** 2)
        return loss

loss_fn = CustomLoss()
output = torch.randn(10, 10)
target = torch.randn(10, 10)
loss = loss_fn(output, target)
print(loss)
