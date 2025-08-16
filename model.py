
import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=27):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flattened_size = None  # будет определено при первом проходе
        self.fc1 = None
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (32, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # -> (64, H/4, W/4)

        if self.flattened_size is None:
            self.flattened_size = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(self.flattened_size, 512)
            # Переместим fc1 на то же устройство, что и x
            self.fc1.to(x.device)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
