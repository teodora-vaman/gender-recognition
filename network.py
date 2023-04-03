import torch
import torch.nn as nn
import torch.utils.data
from icecream import ic



class CNN(nn.Module):
    def __init__(self, nr_clase, image_shape):
        super(CNN, self).__init__()

        # self.conv = nn.Sequential(
        #     # input: nr_imag x 3 x 128 x 128
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride = [1, 1], padding = [1, 1]),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     # nr_imag x 64 x 64 x 64
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride = [1, 1], padding = [1, 1]),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=[2,2], stride=[2,2]),

        #     # nr_imag x 128 x 32 x 32
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride = [1, 1], padding = [1, 1]),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     # nr_imag x 256 x 16 x 16
        # )
        width = image_shape[1]
        height = image_shape[2]

        # input: nr_imag x 3 x 128 x 128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride = [1, 1], padding = [1, 1])
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # nr_imag x 64 x 64 x 64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride = [1, 1], padding = [1, 1])
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=[2,2], stride=[2,2])

        # nr_imag x 128 x 32 x 32
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride = [1, 1], padding = [1, 1])
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # nr_imag x 256 x 16 x 16

        self.fully_connected = nn.Sequential(
            nn.Linear(in_features= 256 * (width // 8) * (height // 8), out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=nr_clase),
            nn.ReLU(),
            nn.Sigmoid()
            )
        
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)
        x = self.fully_connected(x)

        return x


