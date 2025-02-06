from torch.utils.data import DataLoader
from coco_pose_dataset import COCOPoseDataset

if __name__ == "__main__":
    # COCO 데이터 경로 예시 (실제 경로에 맞춰 변경)
    img_dir = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\train2017"
    ann_file = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\annotations\person_keypoints_train2017.json"

    dataset = COCOPoseDataset(img_dir=img_dir, ann_file=ann_file)
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 첫 번째 배치를 가져와 shape만 확인
    for images, heatmaps in dataloader:
        print("배치 이미지 텐서 shape:", images.shape)     
        print("배치 Heatmap 텐서 shape:", heatmaps.shape)
        break
