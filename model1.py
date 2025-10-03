import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CustomNet(nn.Module):
    def __init__(self):
        super(CIFAR10CustomNet, self).__init__()
        self.drop_prob = 0.01

        # Block 1: 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # RF: 3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # RF: 7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.drop_prob)
        )

        # Transition to 64 channels
        self.trans1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Block 2: 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # RF: 11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # RF: 19
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.drop_prob)
        )

        # Transition to 128 channels
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Block 3: 8x8 -> 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # RF: 23
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # RF: 35
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.drop_prob)
        )

        # Transition to 256 channels
        self.trans3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Block 4: 4x4 -> 2x2
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # RF: 39
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # RF: 61
        )

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.trans3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)
