import torch
import torch.nn as nn
import torchvision.models as models

class ResNetPoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(ResNetPoseNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # ✅ 최신 방식으로 가중치 로드
        self.backbone.fc = nn.Linear(512, num_keypoints * 2)  # ✅ 17개 관절 * (x, y)

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 17, 2)  # (batch, num_keypoints, 2)
