import torch
import torch.distributions as D
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from tensordict import TensorDict

from utils import SAVED_POLICIES, SAVED_POLICIES_2V1
from football_design import FootballDesign
from simulate_policy import setup_environment, make_env
from mappo_vmas_training import build_mappo_modules
from plotting import plot_tsne_clusters, plot_snd_heatmap



class MAPPOConfig:
    # Environment
    max_steps = 500
    scenario_name = "football"
    scenario = FootballDesign
    b_agents = 2
    r_agents = 1
    n_agents = b_agents + r_agents
    observe_teammates = False

    # Model
    mappo = True
    share_parameters_policy = b_agents == 1
    share_parameters_critic = b_agents == 1
    num_cells = 256
    depth = 2



def get_latent_features(policy, states, agent_key):
    """Extracts activations from first hidden layer of the MLP."""
    mlp = policy.module[0]
    inner_mlp = next(mlp.children())
    feature_extractor = torch.nn.Sequential(inner_mlp[0], inner_mlp[1])

    # extract latent representation of policies
    with torch.no_grad():
        obs = states[(agent_key, "observation")]
        latents = feature_extractor(obs)
    
    latents = latents.reshape(-1, latents.shape[-1])
    return latents.cpu().numpy()


def setup_and_get_policies(config, checkpoint_path_a, checkpoint_path_b, a_gnn=False, b_gnn=False):
    """Setup the environment for evaluation and get both policies from saved policy path names."""
    device, vmas_device = setup_environment(seed=0)
    env, agent_key = make_env(config, vmas_device)

    policy_a, critic_a = build_mappo_modules(env, config, device, agent_key, a_gnn)
    checkpoint_a = torch.load(checkpoint_path_a, map_location=device)
    policy_a.load_state_dict(checkpoint_a['policy_state_dict'])
    critic_a.load_state_dict(checkpoint_a['critic_state_dict'])
    policy_a.eval()
    critic_a.eval()

    policy_b, critic_b = build_mappo_modules(env, config, device, agent_key, b_gnn)
    checkpoint_b = torch.load(checkpoint_path_b, map_location=device)
    policy_b.load_state_dict(checkpoint_b['policy_state_dict'])
    critic_b.load_state_dict(checkpoint_b['critic_state_dict'])
    policy_b.eval()
    critic_b.eval()

    print("LOADED BOTH POLICIES.")
    return env, policy_a, policy_b, critic_a, critic_b, agent_key, device


def calculate_snd(policy_a, policy_b, states, agent_key):
    """Calculates System Neural Diversity (SND) via KL divergence between two 1v1 policies on the same batch of states."""
    states_a = states.clone()
    states_b = states.clone()

    with torch.no_grad():
        td_a = policy_a(states_a)
        loc_a = td_a[agent_key, "loc"][:, 0, :].clone()
        scale_a = td_a[agent_key, "scale"][:, 0, :].clone()

        td_b = policy_b(states_b)
        loc_b = td_b[agent_key, "loc"][:, 0, :].clone()
        scale_b = td_b[agent_key, "scale"][:, 0, :].clone()

        # normal distributions for policies a and b to calculate kl divergence
        dist_a = D.Normal(loc_a, scale_a)
        dist_b = D.Normal(loc_b, scale_b)
        kl_ab = D.kl_divergence(dist_a, dist_b).sum(dim=-1)
        kl_ba = D.kl_divergence(dist_b, dist_a).sum(dim=-1)
        kl = ((kl_ab + kl_ba) / 2).mean().item()


    return kl_ab.mean().item(), kl_ba.mean().item(), kl


def calculate_snd_wasserstein(policy_a, policy_b, states, agent_key):
    """Calculates SND via 2-Wasserstein Distance for Gaussian policies."""
    with torch.no_grad():
        td_a = policy_a(states)
        mu_a = td_a[agent_key, "loc"] # mean [Samples, Agents, Action_dim]
        std_a = td_a[agent_key, "scale"]

        td_b = policy_b(states)
        mu_b = td_b[agent_key, "loc"]
        std_b = td_b[agent_key, "scale"]

        mu_a, std_a = mu_a[:, 0, :], std_a[:, 0, :]
        mu_b, std_b = mu_b[:, 0, :], std_b[:, 0, :]

        # 2-Wasserstein Distance for Diagonal Gaussians
        # W^2 = ||mu_a - mu_b||^2 + ||std_a - std_b||^2
        mean_diff = torch.sum((mu_a - mu_b) ** 2, dim=-1)
        std_diff = torch.sum((std_a - std_b) ** 2, dim=-1)
        
        wasserstein_sq = mean_diff + std_diff
        wasserstein_dist = torch.sqrt(wasserstein_sq + 1e-8) # numerical stability with error

        snd_val = wasserstein_dist.mean().item()

    return snd_val


def eval_conjecture_one(env, policy_a_name, policy_b_name, policy_a, policy_b, agent_key, save_plot):
    """Evaluation and evidence for conjecture 1. Calculate SND scores and cluster latent representions by calling plotting tsne code."""
    plot_clusters = False
    env.set_seed(0)
    test_td = env.rollout(max_steps=100)
    test_td = test_td.reshape(-1)
    print("GENERATED ROLLOUT SAMPLES.")

    snd_val = calculate_snd_wasserstein(policy_a, policy_b, test_td, agent_key)
    print(f"Symmetric System Neural Diversity (SND) between A ({policy_a_name}) and B ({policy_b_name}): {snd_val:.4f}")

    if plot_tsne_clusters:
        print("PLOTTING TSNE CLUSTERS.")
        latents_a = get_latent_features(policy_a, test_td, agent_key)
        latents_b = get_latent_features(policy_b, test_td, agent_key)
        plot_tsne_clusters(latents_a, latents_b, save_plot, policy_a_name, policy_b_name)

    return snd_val


def calculate_collective_sufficiency(critic_base, critic_comp, states, agent_key):
    """Calculates monolithic and baseline integral for the collective sufficiency and covering idea proposed in conjecture 2."""
    with torch.no_grad():
        v_base_raw = critic_base(states).get((agent_key, "state_value")) # [1250, 2, 1]
        v_comp_raw = critic_comp(states).get((agent_key, "state_value"))

        v_base_all = v_base_raw.squeeze(-1)
        v_comp_all = v_comp_raw.squeeze(-1)

        v_base = v_base_all.mean(dim=1)
        int_v_base = torch.sum(v_base)

        v_a, v_b = v_comp_all[:, 0], v_comp_all[:, 1]
        v_a_norm = v_a / torch.sum(v_a)
        v_b_norm = v_b / torch.sum(v_b)

        v_union = torch.max(v_a_norm, v_b_norm) # take the max normalised utility at each state
        v_union_scaled = v_union * int_v_base

        mse = torch.mean((v_base - v_union_scaled) ** 2) # mean squared difference between value functions
        sufficiency_ratio = v_union_scaled / int_v_base # higher the better, or close to 1.0 means matching [1250]

    return mse.item(), sufficiency_ratio.mean().item(), v_base.mean().item(), v_union_scaled.mean().item()


def eval_conjecture_two(env, policy_base, policy_comp, agent_key):
    """Evaluation and evidence for conjecture 2. Collective sufficiency and composed team comparison to monolithic baseline."""
    print("Evaluating Conjecture 2")

    device = next(policy_base.parameters()).device
    pitch_length, pitch_width = env.scenario.pitch_length, env.scenario.pitch_width

    x = torch.linspace(-pitch_length / 2, pitch_length / 2, 50)
    y = torch.linspace(-pitch_width / 2, pitch_width / 2, 25)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    ball_positions = torch.stack([grid_x.flatten(), grid_y.flatten()], axis=1).to(device)
    num_samples = len(ball_positions)

    obs_spec = env.observation_spec[agent_key, "observation"]
    n_agents, obs_dim = obs_spec.shape[1], obs_spec.shape[2]
    obs_tensor = torch.zeros((num_samples, n_agents, obs_dim), device=device)

    for i in range(num_samples): obs_tensor[i, :, 2:4] = -ball_positions[i]
    states = TensorDict({(agent_key, "observation"): obs_tensor}, batch_size=[num_samples], device=device) # [1250, 1, 2]

    mse, suff_ratio, v_base, v_union = calculate_collective_sufficiency(policy_base, policy_comp, states, agent_key)

    print(f"MSE between baseline and composed policy: {mse:.4f}")
    print(f"V_base: {v_base:.4f}, V_union: {v_union:.4f}")
    print(f"Sufficiency Ratio (Z-scaled): {suff_ratio:.4f}")

    return mse, suff_ratio, v_base, v_union


def run_evaluation_episodes(env, policy, num_episodes, agent_key, device):
    """Runs a set number of episodes and returns average reward for conjecture 3 evaluation."""
    """
    total_rewards = []

    policy.eval()
    policy.to(device)

    for _ in range(num_episodes):
        td = env.reset()
        episode_reward = 0

        with torch.no_grad():
            for _ in range(env.scenario.max_steps):
                with set_exploration_type(ExplorationType.DETERMINISTIC): td = policy(td)

                td = env.step(td)
                reward = td.get(("next", agent_key, "reward")).mean()
                episode_reward += reward.item()

                if td.get(("next", "done")).any(): break
                td = step_mdp(td)
        
        total_rewards.append(episode_reward)

    return torch.tensor(total_rewards).mean().item()
    """


def eval_conjecture_three(env, policy_base, policy_spec, agent_key, device, num_eps=50):
    """Evaluation and evidence for conjecture 3. Cumulative reward comparison between specialised teams comparison to monolithic baseline."""
    """
    print(f"Evaluating Conjecture 3 for {num_eps} episodes")

    print("Running Baseline Evaluation")
    reward_base = run_evaluation_episodes(env, policy_base, num_eps, agent_key, device)
    print("Running Team Evaluation")
    reward_spec = run_evaluation_episodes(env, policy_spec, num_eps, agent_key, device)

    epsilon = 0.05 * reward_base # define error tolerance of x% of baseline
    difference = reward_spec - reward_base
    success = reward_spec >= (reward_base - epsilon)

    print("-" * 30)
    print(f"baseline mean cumulative reward: {reward_base:.4f}")
    print(f"specialised team mean cumulative reward: {reward_spec:.4f}")
    print(f"performance diffeerence: {difference:.4f}")
    print(f"conjecture 3 validatied: {success}")
    print("-" * 30)

    return reward_base, reward_spec, success
    """


if __name__ == "__main__":
    config = MAPPOConfig()
    EVAL_CONJECTURE_1 = False
    EVAL_CONJECTURE_2 = True
    EVAL_CONJECTURE_3 = False

    if EVAL_CONJECTURE_1:
        policy_dict = SAVED_POLICIES
        a_gnn, b_gnn = False, False
        config.b_agents = 1
    else:
        config.b_agents = 2
        a_gnn, b_gnn = False, True
        policy_dict = SAVED_POLICIES_2V1

    policy_a_name, policy_b_name = "baseline", "baseline_vs_mask_rhs_mappo"
    checkpoint_path_a, checkpoint_path_b = policy_dict[policy_a_name], policy_dict[policy_b_name]
    env, policy_a, policy_b, critic_a, critic_b, agent_key, device = setup_and_get_policies(config, checkpoint_path_a, checkpoint_path_b, a_gnn, b_gnn)

    # comparing system neural diversity (SND) and KL divergence between baseline and specialised policies
    if EVAL_CONJECTURE_1:
        snd_val = eval_conjecture_one(env, policy_a_name, policy_b_name, policy_a, policy_b, agent_key, save_plot=True)
    elif EVAL_CONJECTURE_2:
        eval_conjecture_two(env, critic_a, critic_b, agent_key)
    elif EVAL_CONJECTURE_3:
        policy_base = None
        policy_spec = None
        eval_conjecture_three(env, policy_base, policy_spec, agent_key, device)