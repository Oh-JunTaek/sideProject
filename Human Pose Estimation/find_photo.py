from pycocotools.coco import COCO

# 어노테이션 파일 로드 (val2017 기준)
ann_file = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\annotations\person_keypoints_val2017.json"
coco = COCO(ann_file)

# 선택한 이미지 ID
image_id = 274460  # 가장 키포인트 많은 이미지 선택

# 이미지 정보 가져오기
img_info = coco.loadImgs(image_id)[0]
image_path = rf"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\val2017\{img_info['file_name']}"

print("테스트할 이미지 경로:", image_path)
