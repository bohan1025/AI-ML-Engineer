import torch.nn as nn


# CNN Model class
class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) class that inherits from PyTorch's nn.Module.
    This class represents a simple CNN model with two convolutional layers, each followed by a ReLU activation and a max pooling layer.

    Attributes:
    height (int): The height of the input image.
    width (int): The width of the input image.
    conv_layers (nn.Sequential): A sequential container of the convolutional layers of the network.

    Methods:
    forward(x): Defines the forward pass of the CNN.
    """

    def __init__(self, height, width):
        super(CNN, self).__init__()
        self.height = height
        self.width = width
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Parameters:
        x (torch.Tensor): The input to the network.

        Returns:
        x (torch.Tensor): The output from the network.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        fc_layers = nn.Sequential(nn.Linear(x.shape[1], 1), nn.Sigmoid())
        x = fc_layers(x)
        return x
