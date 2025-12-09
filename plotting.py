import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from tensordict.tensordict import TensorDict

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



def make_env(config, vmas_device):
    """Creates VMAS environment and gets agent key."""
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
    """Creates a TensorDict template where all state components except the ball's position are fixed to a canonical reference e.g. zero velocity, agents at origin."""
    state_td = env.reset(inplace=True).clone()
    state_td.get(agent_key)["velocity"].zero_()
    state_td.get(agent_key)["force"].zero_()
    state_td["Ball"]["velocity"].zero_()
    state_td["Ball"]["force"].zero_()
    
    state_td.get(agent_key)["position"].zero_()

    adv_key = f"agent_red_0" 
    if adv_key in state_td.keys():
        state_td.get(adv_key)["position"] = torch.tensor([1.0, 0.0], device=device)
        state_td.get(adv_key)["velocity"].zero_()
        state_td.get(adv_key)["force"].zero_()
        
    return state_td


def run_inference(config, checkpoint_path, grid_points):
    """Loads policy and critic and generates the value and action plots."""
    print(f"Starting plotting. Loading policy from: {checkpoint_path}")

    device, vmas_device = setup_environment()
    env, agent_key = make_env(config, vmas_device)
    policy, critic = build_mappo_modules(env, config, device, agent_key)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    policy.eval()
    critic.eval()
    
    pitch_length, pitch_width = env.scenario.pitch_length, env.scenario.pitch_width
    pitch_x_range, pitch_y_range = (-pitch_length / 2, pitch_length / 2), (-pitch_width / 2, pitch_width / 2)
    x_coords = np.linspace(pitch_x_range[0], pitch_x_range[1], grid_points)
    y_coords = np.linspace(pitch_y_range[0], pitch_y_range[1], grid_points)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    ball_positions = torch.tensor(np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1), dtype=torch.float32, device=device)
    num_samples = len(ball_positions)

    fixed_base_vector = torch.zeros(num_samples, 2, device=device)
    adv_pos_fixed = torch.tensor([1.0, 0.0], device=device).unsqueeze(0).expand(num_samples, 2) 
    goal_pos_fixed = env.scenario.right_goal_pos.clone().unsqueeze(0).expand(num_samples, 2)
    
    obs_batch = env.scenario.observation_base(
        agent_pos=fixed_base_vector, agent_vel=fixed_base_vector, agent_force=fixed_base_vector,
        agent_rot=torch.zeros(num_samples, 1, device=device), # rotation is only an angle
        adversary_poses=[adv_pos_fixed], adversary_forces=[fixed_base_vector], adversary_vels=[fixed_base_vector],
        teammate_poses=[], teammate_forces=[], teammate_vels=[],
        ball_pos=ball_positions, ball_vel=fixed_base_vector, ball_force=fixed_base_vector,
        goal_pos=goal_pos_fixed, 
        blue=True,
    ).unsqueeze(1)

    eval_td = TensorDict({
        agent_key: TensorDict({'observation': obs_batch}, batch_size=[num_samples, 1])
    }, batch_size=[num_samples], device=device)

    pitch_geometry = {
        'X_grid': X_grid, 'Y_grid': Y_grid,
        'X_range': pitch_x_range, 'Y_range': pitch_y_range,
        'agent_key': agent_key
    }

    return eval_td, agent_key, pitch_geometry, policy, critic


def plot_value_heatmap(pitch_geometry, grid_points, eval_td, agent_key, critic, save_path, save):
    """Plots a heatmap of the football pitch from the value function. We use the critic to get the value of states."""
    with torch.no_grad():
        value_td = critic(eval_td)
        V_s = value_td.get((agent_key, 'state_value')).squeeze().cpu().numpy()
    V_s_grid = V_s.reshape(grid_points, grid_points)

    plt.figure(figsize=(13, 6))
    ax = plt.gca()
    norm = Normalize(vmin=V_s_grid.min(), vmax=V_s_grid.max())

    X_range, Y_range = pitch_geometry['X_range'], pitch_geometry['Y_range']
    im = ax.imshow(V_s_grid, origin='lower', 
                   extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
                   aspect='auto', cmap='Greens', norm=norm)
    
    plt.colorbar(im, ax=ax, label='Value Function')
    ax.axvline(x=0, color='lightgrey', linestyle='-')
    
    ax.set_title(f'Pitch value function heatmap.')
    ax.set_xlabel('X-Coordinate')
    ax.set_ylabel('Y-Coordinate')
    
    plt.tight_layout()
    if save:
        save_path = f"{save_path}_val_heatmap.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_action_vectors(pitch_geometry, grid_points, eval_td, policy, critic, agent_key, subsample=5):
    """Plots the policy mean action on the pitch like a vector field."""
    with torch.no_grad():
        value_td = critic(eval_td)
        V_s = value_td.get((agent_key, 'state_value')).squeeze().cpu().numpy()
        # for each state also get the policy for plotting an action vector
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            policy_td = policy(eval_td)
            Mu_x = policy_td.get((agent_key, 'loc'))[..., 0].cpu().numpy()
            Mu_y = policy_td.get((agent_key, 'loc'))[..., 1].cpu().numpy()

    V_s_grid = V_s.reshape(grid_points, grid_points)
    Mu_x_grid = Mu_x.reshape(grid_points, grid_points)
    Mu_y_grid = Mu_y.reshape(grid_points, grid_points)
    
    X_grid, Y_grid = pitch_geometry['X_grid'], pitch_geometry['Y_grid']
    X_range, Y_range = pitch_geometry['X_range'], pitch_geometry['Y_range']
    agent_key = pitch_geometry['agent_key']
    
    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.imshow(V_s_grid, origin='lower', 
               extent=[X_range[0], X_range[1], Y_range[0], Y_range[1]], 
               aspect='auto', cmap='Greens', alpha=0.7)
    
    ax.quiver(X_grid[::subsample, ::subsample], Y_grid[::subsample, ::subsample], 
               Mu_x_grid[::subsample, ::subsample], Mu_y_grid[::subsample, ::subsample], 
               scale=10, scale_units='x', color='red', alpha=0.8)
    
    ax.set_title(f'Policy Mean Action $\\mu(s)$ Vector Field')
    ax.set_xlabel('Pitch X-Coordinate')
    ax.set_ylabel('Pitch Y-Coordinate')
    ax.axvline(x=0, color='lightgrey', linestyle='-')
    
    plt.suptitle(f"Policy Action Analysis for {agent_key}")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    config = MAPPOConfig()
    GRID_POINTS = 200
    PLOT_VALUE_HEATMAP = True
    PLOT_ACTION_VECTORS = False

    checkpoint, policy_no = "011225_195207", "499"
    checkpoint_path = f"./saved_policies/mappo_football_{checkpoint}/iteration_{policy_no}_policy.pt"
    save_path = f"plots/{checkpoint}_{policy_no}"

    eval_td, agent_key, pitch_geometry, policy, critic = run_inference(
        config=config,
        checkpoint_path=checkpoint_path,
        grid_points=GRID_POINTS)

    if PLOT_VALUE_HEATMAP: plot_value_heatmap(pitch_geometry, GRID_POINTS, eval_td, agent_key, critic, save_path, save=False)
    if PLOT_ACTION_VECTORS: plot_action_vectors(pitch_geometry, GRID_POINTS, eval_td, policy, critic, agent_key, save=False)