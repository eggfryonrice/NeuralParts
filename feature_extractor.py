import torch
import torch.nn as nn
from torchvision import models

#feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self, n_feature, n_primitive):
        # feature encoder
        super().__init__() 
        self.n_feature = n_feature
        #print("n_feature is " + str(n_feature))
        self.n_primitive = n_primitive
        self.fe = models.resnet18(pretrained=True)
        self.fc1 = torch.nn.Linear(512, 512)
        self.fc2 = torch.nn.Linear(512, self.n_feature)
        self.fe.fc = nn.Sequential(self.fc1, self.fc2)
        self.fe.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # learnable primitie embedding
        self.register_parameter(name='p', param=torch.nn.Parameter(torch.randn(n_primitive, n_feature)))
    
    def forward(self, x):
        F = self.fe(x)
        #print(self.p)
        F = torch.unsqueeze(F, 1)
        F = F.expand(-1, self.n_primitive, -1) # should have size 4 * 5 * 256
        
        P = torch.unsqueeze(self.p, 0)
        P = P.expand(F.shape[0], -1, -1) # should have size 4 * 5 * 256

        C = torch.cat([F,P], dim=-1)

        return C