import torch
from simple_pose_net import SimplePoseNet

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = SimplePoseNet(num_keypoints=17).to(device)
try:
    model.load_state_dict(torch.load("pose_model.pth", map_location=device))
    print("✅ 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print("❌ 모델 로드 실패:", e)

# 가중치 확인 (일부 레이어 값 출력)
for name, param in model.named_parameters():
    print(f"{name}: {param.mean().item():.6f}")
    break  # 너무 길어지지 않도록 첫 번째 레이어만 출력