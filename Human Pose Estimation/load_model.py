import torch
from simple_pose_net import SimplePoseNet

# 디바이스 설정 (GPU 사용 가능하면 CUDA, 없으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 생성 후 저장된 가중치 로드
model = SimplePoseNet(num_keypoints=17).to(device)
model.load_state_dict(torch.load("pose_model.pth", map_location=device))
model.eval()  # 모델을 평가 모드로 설정

print("✅ Model loaded successfully and ready for inference!")
