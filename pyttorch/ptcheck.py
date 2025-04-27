# import torch
# print(torch.__version__)
# print(torch.version.cuda)

import torch
import torchvision
import torchaudio

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device name: Shrek GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")