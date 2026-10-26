import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, classes=None):
        super().__init__()
        self.num_classes = classes
        self.prepare_net()

    def prepare_net(self):
        self.conv_network = nn.Sequential(
            nn.Conv2d(1, 2, 3, 1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()

        self.linear_out = nn.Sequential(
            nn.LazyLinear(self.num_classes),
        )

    def forward(self, x):
        x = self.conv_network(x)
        x = self.flatten(x)
        x = self.linear_out(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


class ConvNetV2(nn.Module):
    def __init__(self, classes=None):
        super().__init__()
        self.num_classes = classes
        self.prepare_net()

    def prepare_net(self):
        self.conv_network = nn.Sequential(
            nn.Conv2d(1, 7, 3, 1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(7, 4, 3, 1),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.flatten = nn.Flatten()

        self.linear_out = nn.Sequential(
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.LazyLinear(self.num_classes),
        )

    def forward(self, x):
        x = self.conv_network(x)
        x = self.flatten(x)
        x = self.linear_out(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)