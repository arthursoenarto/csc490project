
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.models as models

class model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = models.resnet34(pretrained=False)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_ftrs, 7)
        )
        
    def forward(self, xb):
        return self.network(xb)