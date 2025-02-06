import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from simple_pose_net import SimplePoseNet

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 모델 불러오기
model = SimplePoseNet(num_keypoints=17).to(device)
model.load_state_dict(torch.load("pose_model.pth", map_location=device))
model.eval()  # 모델을 평가 모드로 설정

# 이미지 전처리 함수
def preprocess_image(image_path, target_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 후 GPU로 이동
    return image

# 예측 수행
def predict_pose(image_path):
    image = preprocess_image(image_path)
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        output = model(image)  # 모델 실행 (출력은 (1, 17, 64, 64) 형태의 Heatmap)

    return output.cpu().numpy()

# 테스트 이미지에 대해 실행
image_path = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\val2017\000000000785.jpg"
heatmaps = predict_pose(image_path)

print("✅ Inference completed! Heatmap shape:", heatmaps.shape)
