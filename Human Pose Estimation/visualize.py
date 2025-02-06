import cv2
import numpy as np
import torch
from inference import predict_pose, preprocess_image
from PIL import Image

# 키포인트를 원본 이미지 위에 그리는 함수
def draw_keypoints(image_path, heatmaps):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR, PIL은 RGB이므로 변환

    num_keypoints = heatmaps.shape[1]
    heatmap_size = heatmaps.shape[2]

    # 이미지 크기에 맞게 키포인트 좌표 변환
    height, width, _ = image.shape
    scale_x = width / heatmap_size
    scale_y = height / heatmap_size

    for kp in range(num_keypoints):
        heatmap = heatmaps[0, kp, :, :]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x = int(x * scale_x)
        y = int(y * scale_y)
        
        print(f"키포인트 {kp}: ({x}, {y})")  # 디버깅 출력

        # 키포인트를 원본 이미지 위에 그리기
        cv2.circle(image, (x, y), 15, (255, 0, 0), -1) # red

    return image

# 테스트 이미지에 대해 실행
image_path = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\val2017\000000000785.jpg"
heatmaps = predict_pose(image_path)
result_image = draw_keypoints(image_path, heatmaps)

# 결과 저장 및 출력
cv2.imwrite("result.jpg", cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
cv2.imshow("Pose Estimation", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
