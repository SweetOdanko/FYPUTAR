import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU()
        )
        self.fc = nn.Linear(256, 1)

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, x1, x2):
        h1 = self.forward_once(x1)
        h2 = self.forward_once(x2)
        diff = torch.abs(h1 - h2)
        scores = self.fc(diff)
        return scores

if __name__ == '__main__':
    model = SiameseNetwork()
    print(model)
