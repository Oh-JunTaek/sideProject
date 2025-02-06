import torch

print("PyTorch version:", torch.__version__)
print("CUDA available?", torch.cuda.is_available())

print(torch.__version__)         # ex) 2.6.0+cu118
print(torch.version.cuda)        # ex) 11.8
print(torch.cuda.is_available()) # True -> GPU 인식됨