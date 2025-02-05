from torch.utils.data import Dataset
import os

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform_func):
        """
        Args:
            image_dir (str): 이미지들이 저장된 폴더 경로.
            transform_func (callable): 이미지를 전처리할 함수 (예: preprocess_image_pil).
        """
        self.image_dir = image_dir
        self.transform_func = transform_func
        # 이미지 파일명 목록을 수집 (확장자가 jpg인 파일들만)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        # 전처리 함수를 통해 이미지 텐서를 얻음
        image_tensor = self.transform_func(image_path)
        # (추후) 어노테이션 정보도 함께 반환하도록 수정 가능
        return image_tensor

# 테스트용 코드:
if __name__ == "__main__":
    dataset = CustomImageDataset(
        image_dir=r"C:\Users\dev\Documents\GitHub\sideProject\Human Pose Estimation\data\train2017",
        transform_func=preprocess_image_pil
    )
    print("데이터셋 길이:", len(dataset))
    sample_tensor = dataset[0]
    print("샘플 이미지 텐서 shape:", sample_tensor.shape)
