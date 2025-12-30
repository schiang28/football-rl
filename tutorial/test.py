import torch

t = torch.tensor([[1,2], [-3,6], [-4, 4]])
mask = torch.tensor([False, True, True]).unsqueeze(-1)

ball_pos = torch.where(mask, torch.tensor([[0,0], [0,0], [0,0]]), t)
print(ball_pos)
#ball_vel = torch.where(mask, ZERO_VEL, ball_vel)
