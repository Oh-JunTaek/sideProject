from PIL import Image
import torchvision.transforms as transforms

def preprocess_image_pil(image_path, target_size=(256, 256)):
    """
    이미지 경로를 받아 target_size 크기로 리사이즈하고, 텐서 변환 및 정규화를 수행합니다.
    
    Args:
        image_path (str): 이미지 파일의 경로.
        target_size (tuple): (width, height) 형태의 원하는 출력 크기.
        
    Returns:
        torch.Tensor: 전처리된 이미지 텐서.
    """
    # 전처리 파이프라인 정의 (여기서는 일반적인 ImageNet 정규화 값 사용)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # 0~1 사이의 값으로 변환 및 (C, H, W) 텐서 형식
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # PIL Image로 읽어오기
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image

# 테스트용 코드 (선택 사항)
if __name__ == "__main__":
    # 테스트용 이미지 경로 (실제 존재하는 이미지 파일명을 사용하세요)
    image_path = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\train2017\000000388255.jpg"
    processed_tensor = preprocess_image_pil(image_path)
    print("전처리된 이미지 텐서 shape:", processed_tensor.shape)
