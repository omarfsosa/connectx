"""
A Convolutional network to evaluate the board configurations.
"""
import torch.nn as nn


class CNN(nn.Module):
    """
    Conv2d network to estimate the return of
    a connectX board.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """Evaluate the input"""
        return self.layers(x)