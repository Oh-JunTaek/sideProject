from pycocotools.coco import COCO

# 어노테이션 파일 로드 (훈련 or 검증 데이터셋 선택)
ann_file = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\annotations\person_keypoints_val2017.json"
coco = COCO(ann_file)

# 모든 이미지 ID 가져오기
img_ids = coco.getImgIds()

# 키포인트 개수 기준으로 정렬
img_keypoints = []
for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    
    # 현재 이미지의 모든 사람에 대해 키포인트 개수 합산
    total_keypoints = sum([ann["num_keypoints"] for ann in anns if "num_keypoints" in ann])
    img_keypoints.append((img_id, total_keypoints))

# 키포인트가 많은 상위 5개 이미지 출력
img_keypoints = sorted(img_keypoints, key=lambda x: x[1], reverse=True)[:5]
print("키포인트 많은 상위 5개 이미지:", img_keypoints)
