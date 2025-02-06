from simple_pose_net import SimplePoseNet
from coco_pose_dataset import COCOPoseDataset
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    # 1) Dataset & DataLoader 구성
    img_dir = r"...\train2017"
    ann_file = r"...\person_keypoints_train2017.json"
    dataset = COCOPoseDataset(img_dir, ann_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2) 모델 인스턴스 생성
    model = SimplePoseNet(num_keypoints=17)
    
    # 3) 첫 번째 배치를 읽고 forward 패스
    images, heatmaps = next(iter(dataloader))
    print("입력 이미지 shape:", images.shape)
    print("정답 Heatmap shape:", heatmaps.shape)

    outputs = model(images)
    print("모델 출력 shape:", outputs.shape)
