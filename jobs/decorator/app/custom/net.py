import torch.nn as nn
import torch.nn.functional as F
import torch

class TumorNet(nn.Module):
    """
    A CNN with:
        one convolutional layer
        one max pooling layer
        two fully connected layers
    """

    def __init__(self):
        super(TumorNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=60 * 60 * 16, out_features=64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        """Operations on x"""
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x
    


# torchvision.models.resnet18(pretrained=True).to(device)
        #for p in  model.parameters():
            # freeze all parameters
        #    p.requires_grad = False
            
        #in_features =  model.fc.in_features
            
        #model.fc = nn.Sequential(
        #        nn.Linear(in_features, 1),
        #        nn.Sigmoid()
        #)    