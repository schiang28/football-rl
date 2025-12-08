import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from tensordict.nn import TensorDictModule

from football_design import FootballDesign
from mappo_vmas_training import build_mappo_modules, setup_environment 



class MAPPOConfig:
    # Environment
    max_steps = 500
    scenario_name = "football"
    scenario = FootballDesign
    b_agents = 1
    r_agents = 1
    n_agents = b_agents + r_agents
    observe_teammates = b_agents > 1

    # Model
    mappo = True
    share_parameters_policy = True
    share_parameters_critic = True
    num_cells = 256
    depth = 2

    # Evaluation
    explore = False


def make_env(config, vmas_device):
    """Creates VMAS environment for visualization specs."""
    if config.scenario_name == "football": scenario = config.scenario()
    else: scenario = config.scenario_name

    env = VmasEnv(
        scenario=scenario,
        num_envs=1,
        continuous_actions=True,
        max_steps=config.max_steps,
        device=vmas_device,
        n_blue_agents=config.b_agents,
        n_red_agents=config.r_agents,
        observe_teammates=config.observe_teammates,
    )

    agent_key = env.action_keys[0][0]
    env = TransformedEnv(env)
    return env, agent_key


def prepare_fixed_state(env, device, agent_key):
    """
    Creates a TensorDict template where all state components EXCEPT the ball's position are fixed to a canonical reference (e.g., zero velocity, agents at origin).
    """
    state_td = env.reset(inplace=True).clone()
    
    # 1. Fix all dynamic variables to zero
    state_td.get(agent_key)["velocity"].zero_()
    state_td.get(agent_key)["force"].zero_()
    state_td["Ball"]["velocity"].zero_()
    state_td["Ball"]["force"].zero_()
    
    # 2. Fix agent position for clear relative observation (Blue agent is at origin)
    state_td.get(agent_key)["position"].zero_()

    # 3. Handle Adversary (Red Agent)
    adv_key = f"agent_red_0" 
    if adv_key in state_td.keys():
        state_td.get(adv_key)["position"] = torch.tensor([1.0, 0.0], device=device) # Fixed adversary pos
        state_td.get(adv_key)["velocity"].zero_()
        state_td.get(adv_key)["force"].zero_()
        
    return state_td


def plot_value(checkpoint_path, grid_points, pitch_x_range, pitch_y_range):
    """Loads policy and critic and generates the Value and Action plots."""
    print(f"Starting plotting. Loading policy from: {checkpoint_path}")
    
    device, vmas_device = setup_environment()
    env, agent_key = make_env(config, vmas_device)
    policy, critic = build_mappo_modules(env, config, device, agent_key)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    
    policy.eval()
    critic.eval()
    
    # Prepare base state template (Agent at (0,0), all speeds zero)
    scenario_obj = env.env.scenario
    right_goal_pos = scenario_obj.right_goal_pos

    blue_agent = env.unwrapped.world.agents[0]
    red_agent = env.unwrapped.world.agents[1] if len(env.unwrapped.world.agents) > 1 else None
    ball = env.unwrapped.world.ball
    
    # --- 1. Create Grid and Batch TD ---
    x_coords = np.linspace(pitch_x_range[0], pitch_x_range[1], grid_points)
    y_coords = np.linspace(pitch_y_range[0], pitch_y_range[1], grid_points)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    ball_positions = torch.tensor(
        np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1),
        dtype=torch.float32,
        device=device
    )
    num_samples = len(ball_positions)
    
    agent_pos_fixed = torch.zeros(num_samples, 2, device=device) # Blue agent fixed at (0, 0)
    agent_vel_fixed = torch.zeros(num_samples, 2, device=device)
    agent_force_fixed = torch.zeros(num_samples, 2, device=device)
    agent_rot_fixed = torch.zeros(num_samples, 1, device=device)
    adv_pos_fixed = torch.tensor([1.0, 0.0], device=device).unsqueeze(0).expand(num_samples, 2) 
    # Repeat the fixed base TD for all grid points

    all_obs_vectors = []
    
    for i in range(num_samples):
        # Generate the observation vector by calling the internal scenario method:
        obs_tensor = env.unwrapped.scenario.observation_base(
            agent_pos=agent_pos_fixed[i],
            agent_rot=agent_rot_fixed[i],
            agent_vel=agent_vel_fixed[i],
            agent_force=agent_force_fixed[i],
            # Pass only the position/velocity of the current ball sample
            ball_pos=ball_positions[i],
            ball_vel=torch.zeros(2, device=device),
            ball_force=torch.zeros(2, device=device),
            goal_pos=env.unwrapped.scenario.right_goal_pos, # Target goal position (fixed)
            blue=True,
            
            # Adversary/Teammate positions must be passed as LISTS of tensors
            adversary_poses=[adv_pos_fixed[i]],
            adversary_forces=[torch.zeros(2, device=device)],
            adversary_vels=[torch.zeros(2, device=device)],
            teammate_poses=[], # No teammates in 1v1
            teammate_forces=[],
            teammate_vels=[],
        )
        all_obs_vectors.append(obs_tensor)

    obs_batch = torch.stack(all_obs_vectors, dim=0)
    eval_td = torch.TensorDict({
        agent_key: torch.TensorDict({'observation': obs_batch}, batch_size=[num_samples])
    }, batch_size=[num_samples], device=device)
    
    with torch.no_grad():
        value_td = critic(eval_td)
        V_s = value_td.get((agent_key, 'state_value')).squeeze().cpu().numpy()
        
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            policy_td = policy(eval_td)
            Mu_x = policy_td.get((agent_key, 'loc'))[..., 0].cpu().numpy()
            Mu_y = policy_td.get((agent_key, 'loc'))[..., 1].cpu().numpy()


    V_s_grid = V_s.reshape(grid_points, grid_points)
    Mu_x_grid = Mu_x.reshape(grid_points, grid_points)
    Mu_y_grid = Mu_y.reshape(grid_points, grid_points)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- Plot 1: Value Function Heatmap ---
    ax1 = axes[0]
    norm = Normalize(vmin=V_s.min(), vmax=V_s.max())
    
    im = ax1.imshow(V_s_grid, origin='lower', 
                    extent=[pitch_x_range[0], pitch_x_range[1], pitch_y_range[0], pitch_y_range[1]], 
                    aspect='auto', cmap='viridis', norm=norm)
    fig.colorbar(im, ax=ax1, label='Predicted Value $V(s)$')
    
    ax1.set_title(f'Value Function $V(s)$ (Ball Position)')
    ax1.set_xlabel('Pitch X-Coordinate')
    ax1.set_ylabel('Pitch Y-Coordinate')
    ax1.axvline(x=0, color='gray', linestyle='--') # Center line
    
    # --- Plot 2: Policy Action Quiver Plot (Vector Field) ---
    ax2 = axes[1]
    
    # Quiver plot uses subsampling to avoid cluttering the plot
    subsample = 5
    ax2.quiver(X_grid[::subsample, ::subsample], Y_grid[::subsample, ::subsample], 
               Mu_x_grid[::subsample, ::subsample], Mu_y_grid[::subsample, ::subsample], 
               scale=10, scale_units='x', color='red', alpha=0.8) # Adjust scale for better visualization
    
    # Overlay the Value Function heatmap for context
    ax2.imshow(V_s_grid, origin='lower', 
               extent=[pitch_x_range[0], pitch_x_range[1], pitch_y_range[0], pitch_y_range[1]], 
               aspect='auto', cmap='Greys', alpha=0.3)
    
    ax2.set_title(f'Policy Mean Action $\\mu(s)$ Vector Field')
    ax2.set_xlabel('Pitch X-Coordinate')
    ax2.set_ylabel('Pitch Y-Coordinate')
    ax2.axvline(x=0, color='gray', linestyle='--')
    
    plt.suptitle(f"Policy and Value Function Analysis (Agent: {agent_key})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    config = MAPPOConfig()
    GRID_POINTS = 50
    PITCH_X_RANGE = (-1.5, 1.5)
    PITCH_Y_RANGE = (-0.75, 0.75)

    checkpoint_path = "./saved_policies/mappo_football_011225_195207/iteration_499_policy.pt"

    plot_value(checkpoint_path=checkpoint_path,
               grid_points=GRID_POINTS,
               pitch_x_range=PITCH_X_RANGE,
               pitch_y_range=PITCH_Y_RANGE)