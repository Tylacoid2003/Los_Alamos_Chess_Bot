import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.relu = nn.LeakyReLU(negative_slope=0.02)

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += residual 
        output = self.relu(output)
        return output

class CNN_Residual_Dual_Head_network(nn.Module):
    def __init__(self, num_residual_blocks=6, out_channel_conv=64):
        super().__init__()
        self.out_channel_conv = out_channel_conv
        self.num_residual_blocks = num_residual_blocks

        self.conv1 = nn.Conv2d(in_channels=11, out_channels=self.out_channel_conv, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.out_channel_conv)
        self.relu = nn.LeakyReLU(negative_slope=0.02)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.out_channel_conv) for _ in range(self.num_residual_blocks)
        ])

        self.conv_policy = nn.Conv2d(in_channels=self.out_channel_conv, out_channels=2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(num_features=2)
        self.linear_policy = nn.Linear(in_features=2*6*6, out_features=1356)

        self.conv_value = nn.Conv2d(in_channels=self.out_channel_conv, out_channels=1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(num_features=1)
        self.linear_value_1 = nn.Linear(in_features=1*6*6, out_features=128)
        self.linear_value_2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for block in self.residual_blocks:
            x = block(x)

        p = self.conv_policy(x)
        p = self.bn_policy(p)
        p = self.relu(p)
        p = torch.flatten(p, start_dim=1)
        policy = self.linear_policy(p)

        v = self.conv_value(x)
        v = self.bn_value(v)
        v = self.relu(v)
        v = torch.flatten(v, start_dim=1)
        v = self.linear_value_1(v)
        v = self.relu(v)
        v = self.linear_value_2(v)
        value = torch.tanh(v)

        return policy, value


