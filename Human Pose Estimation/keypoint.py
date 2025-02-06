from pycocotools.coco import COCO
import os

# 예: COCO 2017 train 어노테이션 JSON 파일
train_annotation_file = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\annotations\person_keypoints_train2017.json"
coco_train = COCO(train_annotation_file)

# 이미지 ID 목록 가져오기
img_ids = coco_train.getImgIds()

# 첫 번째 이미지 정보 로드
img_info = coco_train.loadImgs(img_ids[0])[0]
print("이미지 파일명:", img_info['file_name'])
print("이미지 ID:", img_info['id'])

# 해당 이미지의 어노테이션(= 사람 키포인트 정보 등) 로드
ann_ids = coco_train.getAnnIds(imgIds=img_info['id'], iscrowd=False)
anns = coco_train.loadAnns(ann_ids)
print("키포인트 정보 예시:", anns)
