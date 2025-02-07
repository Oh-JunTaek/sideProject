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
    [396.91986, 125.40627], [384.28397, 113.19349], [357.68848, 108.36974],
    [378.0459, 100.082695], [147.31769, 72.80783], [439.58835, 154.01808],
    [393.80966, 156.50447], [448.84732, 205.02428], [374.3959, 203.47243],
    [439.41943, 224.10767], [366.96207, 216.2085], [441.38068, 272.10345],
    [408.06808, 271.70392], [398.31348, 299.77182], [371.58908, 297.96512],
    [379.52823, 351.85037], [355.95764, 350.56714]
])

# 원본 이미지 로드 
image_path = "C:/Users/dev/Documents/GitHub/sideProject/Human Pose Estimation/data/val2017/000000000785.jpg"
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
