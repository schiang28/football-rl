import torch
import imageio, os
import datetime
import time


from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv

from football_design import FootballDesign


def record_rollout(policy, config, device, gif_path):
    """Runs a single episode rollout using policy and saves it as a GIF."""
    record_env = VmasEnv(
        scenario=config.scenario(),
        num_envs=1,
        continuous_actions=True,
        max_steps=config.max_steps,
        device=device,
        n_agents=config.n_agents,
        render_mode="rgb_array",
    )

    record_env = TransformedEnv(record_env) 
    all_frames = []

    with torch.no_grad():
        td = record_env.reset()
        dones = td.get("done", torch.zeros(1,1, dtype=torch.bool, device=device))
        
        for _ in range(config.max_steps - 1):
            if dones.all():
                print("episode ended early")
                break

            td = policy(td) 
            td = record_env.step(td)
            frame = record_env.render(mode="rgb_array")
            all_frames.append(frame)
            dones = td.get("done", torch.zeros(1,1, dtype=torch.bool, device=device))
            
        record_env.close()

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, all_frames, fps=30)
    print(f"rollout gif saved to {gif_path}")