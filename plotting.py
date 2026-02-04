import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from tensordict.tensordict import TensorDict

from football_design import FootballDesign
from simulate_policy import build_mappo_modules, setup_environment 



class MAPPOConfig:
    # Environment
    max_steps = 500
    scenario_name = "football"
    scenario = FootballDesign
    b_agents = 1
    r_agents = 1
    n_agents = b_agents + r_agents
    observe_teammates = False

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


def run_inference(config, checkpoint_path, grid_points):
    """Loads policy and critic and generates the value and action plots."""
    print(f"Starting plotting. Loading policy from: {checkpoint_path}")

    device, vmas_device = setup_environment(seed=0)
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
    fixed_rot_vector = torch.zeros(num_samples, 1, device=device)
    adv_pos_fixed = torch.tensor([1.0, 0.0], device=device).unsqueeze(0).expand(num_samples, 2) 
    goal_pos_fixed = env.scenario.right_goal_pos.clone().unsqueeze(0).expand(num_samples, 2)
    # all shape [40000, 2]
    
    obs_batch = env.scenario.observation_base(
        agent_pos=fixed_base_vector, agent_vel=fixed_base_vector, agent_force=fixed_base_vector, agent_rot=fixed_rot_vector,
        adversary_poses=[adv_pos_fixed], adversary_forces=[fixed_base_vector], adversary_vels=[fixed_base_vector],
        teammate_poses=[], teammate_forces=[], teammate_vels=[],
        ball_pos=ball_positions, ball_vel=fixed_base_vector, ball_force=fixed_base_vector,
        goal_pos=goal_pos_fixed, 
        blue=True, agent_index=0 # only one agent for plotting so just manually define
    ).unsqueeze(1)

    eval_td = TensorDict({
        agent_key: TensorDict({'observation': obs_batch}, batch_size=[num_samples, 1])
    }, batch_size=[num_samples], device=device)

    pitch_geometry = {
        'X_grid': X_grid, 'Y_grid': Y_grid,
        'X_range': pitch_x_range, 'Y_range': pitch_y_range,
        'agent_key': agent_key,
        'goal_depth': env.scenario.goal_depth,
        'goal_size': env.scenario.goal_size
    }

    return eval_td, agent_key, pitch_geometry, policy, critic


def plot_value_heatmap(pitch_geometry, grid_points, eval_td, agent_key, critic, title, save_path, save):
    """Plots a heatmap of the football pitch from the value function. We use the critic to get the value of states."""
    with torch.no_grad():
        value_td = critic(eval_td)
        V_s = value_td.get((agent_key, 'state_value')).squeeze().cpu().numpy()
    V_s_grid = V_s.reshape(grid_points, grid_points)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    norm = Normalize(vmin=V_s_grid.min(), vmax=V_s_grid.max())

    X_range, Y_range = pitch_geometry['X_range'], pitch_geometry['Y_range']
    goal_depth, goal_size = pitch_geometry['goal_depth'], pitch_geometry['goal_size']
    xmin, xmax, ymin, ymax = X_range[0], X_range[1], Y_range[0], Y_range[1]

    im = ax.imshow(V_s_grid, origin='lower', extent=[xmin, xmax, ymin, ymax], aspect='auto', cmap='Greens', norm=norm)
    ax.axvline(x=0, color='lightgrey', linestyle='-')

    left_goal = Rectangle(
        (xmin, -goal_size / 2),
        width=-goal_depth,
        height=goal_size,
        edgecolor='lightgrey',
        facecolor='grey',
        alpha=0.6,
        linewidth=2,
        zorder=10,
        clip_on=False
    )
    ax.add_patch(left_goal)

    right_goal= Rectangle(
        (xmax, -goal_size / 2),
        width=goal_depth,
        height=goal_size,
        edgecolor='lightgrey',
        facecolor='grey',
        alpha=0.6,
        linewidth=2,
        zorder=10,
        clip_on=False
    )
    ax.add_patch(right_goal) 

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgrey')
        spine.set_linewidth(2)

    plt.colorbar(im, ax=ax, label='Value Function')
    ax.set_title(f'Pitch value function heatmap for {title}')
    
    plt.tight_layout()
    if save:
        save_path = f"{save_path}_val_heatmap.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()

    return V_s_grid


def plot_value_profile(heatmap_data, pitch_geometry, title, save_path, save):
    """Integrates value function from the heatap data along the y-axis for quantification."""
    xmin, xmax = pitch_geometry['X_range']
    ymin, ymax = pitch_geometry['Y_range']
    dy = (ymax - ymin) / heatmap_data.shape[0]
    integrated_profile = np.sum(heatmap_data, axis=0) * dy
    x_bins = np.linspace(xmin, xmax, len(integrated_profile))
    
    # x_indices = np.arange(len(value_profile))
    # slope, intercept = np.polyfit(x_indices, value_profile, 1)
    # print(f"Goal Gravity (Linear Slope): {slope:.6f}")
    # plt.plot(x_bins, slope * x_indices + intercept, '--', color='red', alpha=0.6, label=f'Goal Gravity (Slope: {slope:.4f})')
    
    plt.figure(figsize=(10, 5))
    plt.axvline(x=0, color='lightgrey', linestyle='-')
    plt.plot(x_bins, integrated_profile, color='green', label='Integral ($\int V dy$)')
    plt.fill_between(x_bins, integrated_profile, color='green', alpha=0.2)

    max_val = np.max(integrated_profile)
    min_val = np.min(integrated_profile)
    plt.axhline(y=max_val, color='forestgreen', linestyle='--', alpha=0.7, label=f'Max Value ({max_val:.3f})')
    plt.axhline(y=min_val, color='darkred', linestyle='--', alpha=0.7, label=f'Min Value ({min_val:.3f})')
    
    plt.title(f"Value Function Profile: {title}")
    plt.xlabel("Pitch X-Coordinate")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save:
        save_path = f"{save_path}_val_profile.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_action_vectors(pitch_geometry, grid_points, eval_td, policy, critic, agent_key, save_path, save, subsample=5):
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
               scale=10, scale_units='x', color='green', alpha=0.7)
    
    ax.set_title(f'Policy Mean Action $\\mu(s)$ Vector Field')
    ax.set_xlabel('Pitch X-Coordinate')
    ax.set_ylabel('Pitch Y-Coordinate')
    ax.axvline(x=0, color='lightgrey', linestyle='-')
    
    plt.suptitle(f"Policy Action Analysis for {agent_key}")
    plt.tight_layout()
    if save:
        save_path = f"{save_path}_val_policymap.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_tsne_clusters(features_a, features_b, save, label_a="baseline", label_b="specialised"):
    """Given two latent representations of features, plot TSNE clusters to show difference in policies."""
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    all_features = np.concatenate([features_a, features_b], axis=0)

    # cluster and reduce features to 2 dimensions
    tsne = TSNE(n_components=2, perplexity=30)
    reduced = tsne.fit_transform(all_features)

    plt.figure(figsize=(8,8))
    
    # plot policy A
    plt.scatter(reduced[:len(features_a), 0], reduced[:len(features_a), 1],
                label=label_a,
                alpha=0.6,
                s=40,
                edgecolor='w',
                linewidth=0.5,
                color=sns.color_palette("muted")[0])

    # plot policy B
    plt.scatter(reduced[len(features_a):, 0], reduced[len(features_a):, 1],
                label=label_b,
                alpha=0.6,
                s=40,
                edgecolor='w',
                linewidth=0.5,
                color=sns.color_palette("muted")[1])

    plt.title(f"t-SNE Projection: policies {label_a} vs {label_b}", pad=20)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    sns.despine()
    plt.tight_layout()

    if save:
        save_path = f"plots/tsne_{label_a}_vs_{label_b}.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_snd_heatmap(data_dict, label_a, label_b, save):
    """Plot matrix using data dict and better than normal."""
    policies = sorted(list(set([k for t in data_dict.keys() for k in t])))
    matrix = pd.DataFrame(index=policies, columns=policies, dtype=float)
    
    for (a, b), val in data_dict.items():
        matrix.loc[a, b] = val
        matrix.loc[b, a] = val
    matrix.fillna(0, inplace=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'SND'})
    
    plt.title(f"SND heatmap of baseline and specialised policies")
    plt.tight_layout()
    
    if save:
        save_path = f"plots/heatmap_{label_a}_vs{label_b}.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()



if __name__ == "__main__":
    config = MAPPOConfig()
    GRID_POINTS = 200
    PLOT_VALUE_HEATMAP = True
    PLOT_ACTION_VECTORS = False

    # script for plotting value heatmaps or action vectors
    checkpoint_id, policy_no = "011225_195207", "499"
    checkpoint_path = f"./saved_policies/mappo_football_{checkpoint_id}/iteration_{policy_no}_policy.pt"
    save_path = f"plots/{checkpoint_id}_{policy_no}"
    title = "1v1 play baseline full observation"

    eval_td, agent_key, pitch_geometry, policy, critic = run_inference(
        config=config,
        checkpoint_path=checkpoint_path,
        grid_points=GRID_POINTS)

    if PLOT_VALUE_HEATMAP:
        heatmap_grid = plot_value_heatmap(pitch_geometry, GRID_POINTS, eval_td, agent_key, critic, title, save_path, save=True)
        plot_value_profile(heatmap_grid, pitch_geometry, title, save_path, save=True)
    if PLOT_ACTION_VECTORS: plot_action_vectors(pitch_geometry, GRID_POINTS, eval_td, policy, critic, agent_key, save_path, save=False)