import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1).to(mps_device)
    print(x)
else:
    print("MPS device not found.")
