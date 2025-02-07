import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class COCOPoseDataset(Dataset):
    def __init__(self, img_dir, ann_file, input_size=(224, 224)):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.input_size = input_size

        # ✅ 이미지 전처리 파이프라인 (224x224 변경)
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

        # ✅ 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        # ✅ 키포인트 로드 (17개 관절, (x, y) 좌표만 추출)
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        keypoints = np.zeros((17, 2))  # 기본값 (0, 0)으로 초기화
        if len(anns) > 0 and 'keypoints' in anns[0]:
            for i in range(17):
                x = anns[0]['keypoints'][i * 3]   # x 좌표
                y = anns[0]['keypoints'][i * 3 + 1]  # y 좌표
                v = anns[0]['keypoints'][i * 3 + 2]  # 가시성 (0=보이지 않음, 1=부분, 2=완전)
                
                if v > 0:  # 가시성이 1 또는 2인 경우만 사용
                    keypoints[i, 0] = x / img_info["width"]   # x 좌표 정규화
                    keypoints[i, 1] = y / img_info["height"]  # y 좌표 정규화

        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)  # 텐서 변환

        return image_tensor, keypoints_tensor
