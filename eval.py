import torch
import torch.distributions as D

from utils import SAVED_POLICIES
from football_design import FootballDesign
from simulate_policy import build_mappo_modules, setup_environment, make_env
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
    share_parameters_policy = True
    share_parameters_critic = True
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


def setup_and_get_policies(config, checkpoint_path_a, checkpoint_path_b):
    """Setup the environment for evaluation and get both policies from saved policy path names."""
    device, vmas_device = setup_environment(seed=0)
    env, agent_key = make_env(config, vmas_device)

    policy_a, _ = build_mappo_modules(env, config, device, agent_key)
    checkpoint_a = torch.load(checkpoint_path_a, map_location=device)
    policy_a.load_state_dict(checkpoint_a['policy_state_dict'])
    policy_a.eval()

    policy_b, _ = build_mappo_modules(env, config, device, agent_key)
    checkpoint_b = torch.load(checkpoint_path_b, map_location=device)
    policy_b.load_state_dict(checkpoint_b['policy_state_dict'])
    policy_b.eval()

    print("LOADED BOTH POLICIES.")
    return env, policy_a, policy_b, agent_key


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


def eval_conjecture_one(env, policy_a_name, policy_b_name, policy_a, policy_b, agent_key, save_plot):
    """Evaluation and evidence for conjecture 1. Calculate SND scores and cluster latent representions by calling plotting tsne code."""
    env.set_seed(0)
    test_td = env.rollout(max_steps=100)
    test_td = test_td.reshape(-1)
    print("GENERATED ROLLOUT SAMPLES.")

    snd_ab, snd_ba, snd_val = calculate_snd(policy_a, policy_b, test_td, agent_key)
    latents_a = get_latent_features(policy_a, test_td, agent_key)
    latents_b = get_latent_features(policy_b, test_td, agent_key)

    print(f"KL divergence AB: {snd_ab}")
    print(f"KL divergence BA: {snd_ba}")
    print(f"Symmetric System Neural Diversity (SND) between A ({policy_a_name}) and B ({policy_b_name}): {snd_val}")

    print("PLOTTING TSNE CLUSTERS.")
    plot_tsne_clusters(latents_a, latents_b, save_plot, policy_a_name, policy_b_name)
    return snd_val



if __name__ == "__main__":
    config = MAPPOConfig()
    EVAL_CONJECTURE_1 = True
    EVAL_CONJECTURE_2 = False
    EVAL_CONJECTURE_3 = False

    policy_a_name, policy_b_name = "mask_lhs", "mask_rhs"
    checkpoint_path_a, checkpoint_path_b = SAVED_POLICIES[policy_a_name], SAVED_POLICIES[policy_b_name]
    env, policy_a, policy_b, agent_key = setup_and_get_policies(config, checkpoint_path_a, checkpoint_path_b)

    # comparing system neural diversity (SND) and KL divergence between baseline and specialised policies
    if EVAL_CONJECTURE_1: snd_val = eval_conjecture_one(env, policy_a_name, policy_b_name, policy_a, policy_b, agent_key, save_plot=False)
    elif EVAL_CONJECTURE_2: pass
    elif EVAL_CONJECTURE_3: pass