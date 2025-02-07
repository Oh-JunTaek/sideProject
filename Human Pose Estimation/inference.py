import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from resnet_pose_net import ResNetPoseNet  # ResNetPoseNet 모델 사용

# 1️⃣ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ 저장된 모델 불러오기
model = ResNetPoseNet(num_keypoints=17).to(device)
model.load_state_dict(torch.load("pose_model_22.pth", map_location=device))  # ✅ 최신 모델 로드
model.eval()  # 모델을 평가 모드로 설정

# 3️⃣ 이미지 전처리 함수
def preprocess_image(image_path, target_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 후 GPU로 이동
    return image

# 4️⃣ 예측 수행
def predict_pose(image_path):
    image = preprocess_image(image_path)
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        output = model(image)  # ✅ (1, 17, 2) 형태의 (x, y) 좌표 예측

    # 결과 텐서를 NumPy 배열로 변환 (0~1 정규화 되어있음)
    keypoints = output.cpu().numpy()[0]  # shape: (17, 2)
    
    return keypoints

# 5️⃣ 예측 결과를 이미지에 시각화
def visualize_keypoints(image_path, keypoints):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    for i, (x, y) in enumerate(keypoints):
        x, y = int(x * w), int(y * h)  # ✅ 0~1 정규화된 좌표를 다시 원본 크기로 변환
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # 초록색 점 표시
        cv2.putText(image, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Predicted Pose", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 6️⃣ 실행 예제
if __name__ == "__main__":
    image_path = "C:/Users/dev/Documents/GitHub/sideProject/Human Pose Estimation/data/train2017/000000000962.jpg"  # 테스트할 이미지 경로
    keypoints = predict_pose(image_path)

    print("✅ Inference completed! Predicted Keypoints:\n", keypoints)
    visualize_keypoints(image_path, keypoints)
