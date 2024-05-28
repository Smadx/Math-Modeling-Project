import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    DQN模型,输入为状态,输出为Q值

    Args:
        - num_blocks:网格大小

    Shape:
        - Input:[B, C, num_blocks, num_blocks]
        - Output:[B, 2*num_blocks*num_blocks]
    """
    def __init__(self, num_blocks):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(1024 * num_blocks * num_blocks, 512)
        self.fc2 = nn.Linear(512, 2 * num_blocks * num_blocks)

    def forward(self, x):
        x = F.silu(self.conv_in(x))
        x = F.silu(self.conv_2(x))
        x = F.silu(self.conv_3(x))
        x = F.silu(self.fc1(self.fl(x)))
        x = self.fc2(x)
        return x