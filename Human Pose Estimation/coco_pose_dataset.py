import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2  # Heatmap 그릴 때 유용

class COCOPoseDataset(Dataset):
    def __init__(self, img_dir, ann_file, input_size=(256, 256)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.input_size = input_size

        # 예시 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.img_dir, img_info['file_name'])

        # 이미지 로드
        image = Image.open(image_path).convert('RGB')

        # 키포인트 로드
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # (간단 예시) 첫 번째 사람만 키포인트 사용
        if len(anns) > 0 and 'keypoints' in anns[0]:
            keypoints = anns[0]['keypoints']
        else:
            # 사람이 없거나 키포인트가 없는 경우
            keypoints = [0]*(17*3)  # 17개 관절 * (x,y,v)

        # Heatmap 생성(옵션): 모델이 Heatmap Regression 방식을 쓸 경우
        # 간단히 64x64 크기로 줄여서 관절 위치 찍기 예시
        heatmap_size = (self.input_size[1]//4, self.input_size[0]//4)
        num_keypoints = 17
        heatmaps = np.zeros((num_keypoints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)

        # 가우시안 대신 원(circle)으로 표시 (예시)
        for kp in range(num_keypoints):
            x = keypoints[kp*3+0]
            y = keypoints[kp*3+1]
            v = keypoints[kp*3+2]
            if v > 0:  # 보이는 관절에만
                # Heatmap 좌표로 스케일
                x_hm = int(x * heatmap_size[1]/img_info['width'])
                y_hm = int(y * heatmap_size[0]/img_info['height'])
                if 0 <= x_hm < heatmap_size[1] and 0 <= y_hm < heatmap_size[0]:
                    cv2.circle(heatmaps[kp], (x_hm, y_hm), 2, 1.0, -1)

        # 이미지 전처리
        image_tensor = self.transform(image)
        heatmaps_tensor = torch.from_numpy(heatmaps)

        return image_tensor, heatmaps_tensor
