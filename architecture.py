"""Model components: CIFAR-friendly ResNet18 encoder and SimCLR projection head."""

from torchvision import models
import torch.nn as nn

def build_encoder():
    model = models.resnet18(weights = None)
    model.fc = nn.Identity() # deactivate classification layer
    # cifar friendly
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
    

class ProjektionHead(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )
    
    def forward(self, x):
        return self.head(x)
    
