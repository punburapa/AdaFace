import torch
print(torch.__version__)
print(torch.cuda.is_available())      # ต้องได้ True
print(torch.cuda.get_device_name(0)) # ต้องได้ NVIDIA GeForce GTX 1650