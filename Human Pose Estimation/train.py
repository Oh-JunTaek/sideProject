import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from coco_pose_dataset import COCOPoseDataset
from torchvision import models

### ✅ ResNet 기반 PoseNet 모델 정의
class ResNetPoseNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super(ResNetPoseNet, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 최신 권장 방식
        self.backbone.fc = nn.Linear(512, num_keypoints * 2)  # 17개 관절 * (x, y)
    
    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 17, 2)  # (batch, num_keypoints, 2)

def get_next_filename(base_name="pose_model", extension=".pth"):
    """
    기존 모델 파일이 존재하면 자동으로 넘버링하여 새로운 파일명 생성
    예: pose_model.pth → pose_model_1.pth → pose_model_2.pth ...
    """
    counter = 1
    new_filename = f"{base_name}{extension}"

    while os.path.exists(new_filename):  # 파일이 존재하면 숫자를 증가
        new_filename = f"{base_name}_{counter}{extension}"
        counter += 1
    
    return new_filename

def train_pose_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### ✅ ResNet 기반 모델 사용
    model = ResNetPoseNet(num_keypoints=17).to(device)

    print("Using device:", device)

    # 데이터셋 및 로더 구성
    img_dir = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\train2017"
    ann_file = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\annotations\person_keypoints_train2017.json"
    train_dataset = COCOPoseDataset(img_dir, ann_file)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4,  # 데이터 로딩 속도 증가
        pin_memory=True  # GPU로 데이터를 옮길 때 속도 향상
    )

    # ✅ 손실 함수 변경 (MSELoss → Smooth L1 Loss)
    criterion = nn.SmoothL1Loss()

    # ✅ 학습률 낮추기 (1e-3 → 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 15
    early_stopping_threshold = 0.0005  # Loss가 이 값보다 작아지면 학습 중단   
    best_loss = float('inf')  # 최저 Loss 저장
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 에포크 시작 시간
        epoch_start_time = time.time()

        # tqdm을 이용해 진행 바 표시
        loader = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for batch_idx, (images, keypoints) in enumerate(loader):
            images = images.to(device)
            keypoints = keypoints.to(device)

            # ✅ Keypoints 값이 모두 0인 경우 (유효한 관절이 없는 경우) 패스
            if torch.all(keypoints == 0):
                continue

            outputs = model(images)  # ✅ ResNet 모델 사용
            loss = criterion(outputs, keypoints)  # (batch, 17, 2)로 매칭

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # tqdm 진행 바에 현재 배치 Loss를 표시 (Postfix)
            loader.set_postfix({"loss": f"{loss.item():.4f}"})

        # 에포크 종료 후 로그
        epoch_loss = running_loss / len(train_loader)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"\n[Epoch {epoch+1}/{num_epochs}] "
              f"Loss: {epoch_loss:.4f}, "
              f"Time: {epoch_duration:.2f}s")

        # 🔹 Early Stopping 조건 확인
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = get_next_filename()
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model improved! Saved as {save_path}")

        if epoch_loss < early_stopping_threshold:
            print(f"🛑 Early Stopping! Loss가 {early_stopping_threshold} 이하로 감소하여 학습 중단.")
            break

    print("Training finished.")

if __name__ == "__main__":
    train_pose_model()