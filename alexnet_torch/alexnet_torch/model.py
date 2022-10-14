import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_class=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # use nn.Sequential to make codes clear
        # 卷积层和池化层
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 全连接层，用到了Dropout随机失活神经元，默认比例为0.5
        # Dropout可以减少参数个数，防止过拟合
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_class),
        )
        if init_weights:
            self._initialize_weight()

    def forward(self, x):
        x = self.features(x)
        # 使用flatten展平后再传入全连接层
        # 也可以使用view
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    # 网络权重初始化，实际上一般不用，pytorch会自动初始化
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 如果该层是卷积层，那么使用kaiming_normal法初始化权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 如果是全连接层，则使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
