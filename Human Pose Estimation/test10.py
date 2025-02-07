import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from resnet_pose_net import ResNetPoseNet  # 모델 파일 import

# COCO Keypoint 연결 정보
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 얼굴 (코-눈, 눈-귀 연결)
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 상체 (어깨-팔꿈치-손목 연결)
    (5, 11), (6, 12), (11, 12),  # 몸통 (어깨-골반 연결)
    (11, 13), (12, 14), (13, 15), (14, 16)  # 다리 (골반-무릎-발목 연결)
]

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
model = ResNetPoseNet(num_keypoints=17).to(device)
model.load_state_dict(torch.load("pose_model_22.pth", map_location=device))  # 최신 모델 로드
model.eval()  # 평가 모드로 설정

# 이미지 전처리 함수
def preprocess_image(image_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 후 GPU로 이동
    return image

# 추론 함수 (Inference)
def predict_pose(image_path):
    image = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(image)  # (1, 17, 2) 형태의 (x, y) 좌표 예측

    keypoints = output.cpu().numpy()[0]  # NumPy 배열 변환
    return keypoints

# 시각화 함수 (Visualization)
def visualize_keypoints(image_path, keypoints):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
    h, w, _ = image.shape

    # 0~1 정규화된 좌표를 원본 이미지 크기로 변환
    keypoints[:, 0] *= w
    keypoints[:, 1] *= h

    # 키포인트 찍기
    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)  # 🟢 초록색 점

    # 관절 연결선 추가
    for (i, j) in COCO_SKELETON:
        pt1 = tuple(keypoints[i].astype(int))
        pt2 = tuple(keypoints[j].astype(int))
        cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # 🔵 파란색 선으로 연결

    # 결과 출력
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Pose Estimation with Skeleton")
    plt.show()

# 실행
if __name__ == "__main__":
    image_path = "C:/Users/dev/Documents/GitHub/sideProject/Human Pose Estimation/data/val2017/15.jpg"
    keypoints = predict_pose(image_path)

    print("✅ Inference completed! Predicted Keypoints:\n", keypoints)
    visualize_keypoints(image_path, keypoints)
