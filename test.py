from vmas import make_env
import torch
import numpy as np
import matplotlib.pyplot as plt

scenario_to_test = 'football' 
NUM_ENVS = 32 

env = make_env(
    scenario=scenario_to_test, 
    num_envs=NUM_ENVS, 
    dict_spaces=True, 
    wrapper="gymnasium_vec",
    terminated_truncated=True,
    n_blue_agents=2,
)
    
'''
for key in env.action_space.keys():
    print(key)
'''

obs, info = env.reset()

# Run a few steps
for step in range(100):

    # 1. Generate random actions (replace with your policy later)
    # Assuming continuous 2D actions: (32, 2) per agent
    actions = {
        'agent_blue_0': torch.randn(NUM_ENVS, 2),
        'agent_blue_1': torch.randn(NUM_ENVS, 2),
    }

    # 2. Step the environment
    obs, reward, terminated, truncated, info = env.step(actions)

    # 3. Render a single frame (from the first environment in the batch)
    # You need to call render on the UNWRAPPED VMAS environment (the core simulator)
    # The base VMAS environment exposes a render method. We typically access it via env.unwrapped

    # NOTE: The exact access method can be tricky through multiple wrappers.
    # This assumes the GymnasiumVectorizedWrapper has an 'unwrapped' attribute that leads to the VMAS core:
    frame = env.unwrapped.render(mode="rgb_array")

    # 'frame' will be a batch of RGB arrays. We take the first one (index 0)
    # VMAS/Gymnasium usually returns an array of shape (num_envs, width, height, channels)
    if step % 20 == 0: # Display every 20 steps
        plt.imshow(frame[0]) # Display the first environment in the batch
        plt.title(f"Football Scenario - Step {step}")
        plt.axis('off')
        plt.show(block=False) # Use block=False to update in a loop, or remove to just show final frame
        plt.pause(0.01)

env.close()
