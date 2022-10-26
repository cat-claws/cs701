import torch.nn.functional as F
from pipeline import resnet_csra
import torch

class CAM(resnet_csra.ResNet_CSRA):

    def __init__(self, num_heads, lam, num_classes):
        super(CAM, self).__init__(num_heads, lam, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        weight1 = torch.add(self.classifier.multi_head[0].head.weight,self.classifier.multi_head[1].head.weight)
        weight2 = torch.add(self.classifier.multi_head[2].head.weight, self.classifier.multi_head[3].head.weight)
        weight = torch.add(weight1, weight2)
        x = F.conv2d(x, weight)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x
