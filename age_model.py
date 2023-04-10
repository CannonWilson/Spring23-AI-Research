import torch.nn as nn
import torch.nn.functional as F


class CustomAgeNetwork(nn.Module):

    """
    Class for my young/old binary
    classifier for use on a 
    modified CelebA dataset.
    """

    def __init__(self):
        super(CustomAgeNetwork, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm2d(num_features = 10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm2d(num_features = 10)
        # Block 2
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(num_features = 20)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(num_features = 20)
        # Block 3
        self.conv5 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, padding='same')
        self.bn5 = nn.BatchNorm2d(num_features = 30)
        self.conv6 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, padding='same')
        self.bn6 = nn.BatchNorm2d(num_features = 30)
        # Block 4
        self.conv7= nn.Conv2d(in_channels=30, out_channels=40, kernel_size=3, padding='same')
        self.bn7 = nn.BatchNorm2d(num_features = 40)
        self.conv8 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, padding='same')
        self.bn8 = nn.BatchNorm2d(num_features = 40)
        # FC layers
        self.fc1 = nn.Linear(1200, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        """
        Network architecture:

        ## Blocks 1-4
        Conv
        Relu
        BN
        Conv
        Relu
        Max Pool
        BN

        ## FC
        FC1
        Relu
        FC2
        Softmax
        """
        # Block 1
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.max_pool2d(F.relu(self.conv2(x)), 2))
        # Block 2
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.max_pool2d(F.relu(self.conv4(x)), 2))
        # Block 3
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.max_pool2d(F.relu(self.conv6(x)), 2))
        # Block 4
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.bn8(F.max_pool2d(F.relu(self.conv8(x)), 2))
        # FC Layers
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x
