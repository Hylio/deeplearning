import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):

    def __init__(self, in_channels, internal_neurous):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=in_channels, out_channels=internal_neurous, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurous, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        self.in_channels = in_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.in_channels, 1, 1)
        return inputs*x
