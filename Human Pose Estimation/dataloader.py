from torch.utils.data import DataLoader
from dataset import CustomImageDataset
from preprocessing import preprocess_image_pil

def create_dataloader(image_dir, batch_size=4, shuffle=True, num_workers=0):
    """
    지정된 이미지 폴더에서 데이터를 로드하는 DataLoader를 생성합니다.
    
    Args:
        image_dir (str): 이미지들이 저장된 폴더 경로.
        batch_size (int): 배치 크기.
        shuffle (bool): 데이터 셔플 여부.
        num_workers (int): 데이터를 로드할 때 사용할 서브 프로세스의 수.
        
    Returns:
        DataLoader: 구성된 PyTorch DataLoader 객체.
    """
    dataset = CustomImageDataset(
        image_dir=image_dir,
        transform_func=preprocess_image_pil
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    # 테스트용 이미지 폴더 경로 (실제 환경에 맞게 수정하세요)
    image_dir = r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\val2017"
    
    # DataLoader 생성
    dataloader = create_dataloader(image_dir, batch_size=4, shuffle=True)
    
    # 첫 번째 배치를 로드하여 shape 확인
    for batch in dataloader:
        print("배치 이미지 텐서 shape:", batch.shape)
        break