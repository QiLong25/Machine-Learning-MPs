import torch.nn as nn
import torch


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()

        self.ch_in = num_channels
        self.sequenceF = nn.Sequential(
            nn.Conv2d(self.ch_in, self.ch_in, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ch_in),
            nn.ReLU(),
            nn.Conv2d(self.ch_in, self.ch_in, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ch_in),
        )
        self.myReLU = nn.ReLU()

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        f = self.sequenceF(x)
        y = self.myReLU(x + f)
        return y

class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()

        self.num_ch = num_channels
        self.num_classes = num_classes
        self.sequence1 = nn.Sequential(
            nn.Conv2d(1, self.num_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block = Block(self.num_ch)
        # self.sequence2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        # )
        self.myAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        # self.sequence3 = nn.Sequential(
        #     nn.Linear(self.num_ch, self.num_classes),
        # )
        self.myLinear = nn.Linear(self.num_ch, self.num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        hidden1 = self.sequence1(x)
        hidden2 = self.block(hidden1)
        # print(hidden2.size())
        hidden3 = self.myAvgPool(hidden2)
        hidden3 = hidden3.view(hidden3.size(0), hidden3.size(1))
        # print(hidden3.size())
        y = self.myLinear(hidden3)
        return y

# x = torch.ones((5, 1, 4, 4))
# model = ResNet(2, 10)
# with torch.no_grad():
#     result = model(x)
#     print(result.size())
