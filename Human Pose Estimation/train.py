# train.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coco_pose_dataset import COCOPoseDataset
from simple_pose_net import SimplePoseNet

def train_pose_model():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimplePoseNet(num_keypoints=17).to(device)

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

    # 모델, 손실함수, 옵티마이저
    model = SimplePoseNet(num_keypoints=17).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 에포크 시작 시간
        epoch_start_time = time.time()

        # tqdm을 이용해 진행 바 표시
        loader = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for batch_idx, (images, heatmaps) in enumerate(loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(images)
            loss = criterion(outputs, heatmaps)

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

    print("Training finished.")
    
    # 학습이 끝난 후 모델 저장
    torch.save(model.state_dict(), "pose_model.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    train_pose_model()
