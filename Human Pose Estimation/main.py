import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(SimplePoseNet, self).__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # 간단한 중간 레이어 추가
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # 최종적으로 각 keypoint에 대한 heatmap 생성 (업샘플링 포함)
        self.deconv = nn.ConvTranspose2d(256, num_keypoints, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        heatmaps = self.deconv(x)
        return heatmaps

# 모델 생성 예시
model = SimplePoseNet(num_keypoints=17)
print(model)
