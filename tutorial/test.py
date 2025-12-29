import torch

t = torch.tensor([[1,2], [-3,6], [-4, 4]])
y = t[..., 0] < 0
print(y)
