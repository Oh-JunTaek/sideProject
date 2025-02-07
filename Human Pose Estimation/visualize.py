import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# COCO Keypoint 연결 정보
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 얼굴 (코-눈, 눈-귀 연결)
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 상체 (어깨-팔꿈치-손목 연결)
    (5, 11), (6, 12), (11, 12),  # 몸통 (어깨-골반 연결)
    (11, 13), (12, 14), (13, 15), (14, 16)  # 다리 (골반-무릎-발목 연결)
]

# 추론 결과 로드
predicted_keypoints = np.array([
    [0.3826, 0.3682], [0.1812, 0.1606], [0.3347, 0.3491], [0.0180, 0.0295], [0.3062, 0.3784],
    [0.3302, 0.4229], [0.2037, 0.4815], [0.4695, 0.5274], [0.3013, 0.5889], [0.5593, 0.5026],
    [0.5258, 0.6143], [0.3738, 0.6182], [0.2957, 0.6592], [0.5781, 0.5009], [0.5668, 0.6504],
    [0.6718, 0.7114], [0.6834, 0.9126]
])

# 원본 이미지 로드 
image_path = "C:/Users/dev/Documents/GitHub/sideProject/Human Pose Estimation/data/train2017/000000000962.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR이므로 RGB로 변환

# 키포인트 그리기
for (x, y) in predicted_keypoints:
    cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)  # 🔵 파란색 점
    
# 관절 연결선 추가
for (i, j) in COCO_SKELETON:
    pt1 = tuple(predicted_keypoints[i].astype(int))
    pt2 = tuple(predicted_keypoints[j].astype(int))
    cv2.line(image, pt1, pt2, (0, 255, 0), 2)  # 🟢 초록색 선으로 연결

# 결과 이미지 출력
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis("off")
plt.title("Pose Estimation Keypoints")
plt.show()
