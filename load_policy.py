import torch
import imageio
import os
import datetime

from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs import TransformedEnv, RewardSum
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from football_design import FootballDesign


class MAPPOConfig:
    scenario_name = "football"
    scenario = FootballDesign
    b_agents = 1
    r_agents = 1
    n_agents = b_agents + r_agents
    observe_teammates = b_agents > 1

    max_steps = 50
    num_cells = 256
    depth = 2
    share_parameters_policy = True

    checkpoint_iteration = 400
    policy_base_dir = "./saved_policies/" 
    policy_folder_name = f"mappo_{scenario_name}_251112_032018"



def setup_environment(device_choice="cpu"):
    """Sets device based on availability."""
    if device_choice == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else: device = torch.device("cpu")

    return device


def build_mappo_modules(env, config, device, agent_key):
    """Recreates the exact same policy network architecture."""
    n_actions_per_agent = env.full_action_spec[env.action_key].shape[-1]
    n_obs_per_agent = env.observation_spec[agent_key, "observation"].shape[-1]
    n_total_agents = env.n_agents 

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=n_obs_per_agent,
            n_agent_outputs=2 * n_actions_per_agent,
            n_agents=n_total_agents,
            centralised=False,
            share_params=config.share_parameters_policy,
            device=device,
            depth=config.depth,
            num_cells=config.num_cells,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(
        policy_net, 
        in_keys=[(agent_key, "observation")], 
        out_keys=[(agent_key, "loc"), (agent_key, "scale")],
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=[(agent_key, "loc"), (agent_key, "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=True,
    )
    return policy


def load_policy_weights(policy, config, device):
    """Loads saved policy weights from the checkpoint file."""
    policy_path = os.path.join(config.policy_base_dir, config.policy_folder_name, f"iteration_{config.checkpoint_iteration}_policy.pt")

    state_dict = torch.load(policy_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    print(f"Successfully loaded policy from: {policy_path}")

    return policy


def run_custom_rollout(policy, config, device, gif_path, custom_start_config=None):
    """Runs a single episode rollout using the policy and saves it as a GIF.  custom_start_config is a dictionary used to set initial conditions. """
    
    scenario_kwargs = {
        "n_blue_agents": config.b_agents,
        "n_red_agents": config.r_agents,
        "observe_teammates": config.observe_teammates,
    }
    
    if custom_start_config:
        scenario_kwargs.update(custom_start_config) 

    eval_env = VmasEnv(
        scenario=config.scenario(),
        num_envs=1,
        continuous_actions=True,
        max_steps=config.max_steps,
        device=device,
        render_mode="rgb_array",
        **scenario_kwargs,
    )

    agent_key = eval_env.action_keys[0][0]
    eval_env = TransformedEnv(eval_env, RewardSum(in_keys=[eval_env.reward_key], out_keys=[(agent_key, "episode_reward")]))
    
    all_frames = []

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = eval_env.reset()
        dones = td.get("done", torch.zeros(1,1, dtype=torch.bool, device=device))
        
        for _ in range(config.max_steps):
            if dones.all():
                print("Episode ended early.")
                break

            td = policy(td)
            td = eval_env.step(td)
            frame = eval_env.render(mode="rgb_array")
            all_frames.append(frame)
            dones = td.get("done", torch.zeros(1,1, dtype=torch.bool, device=device))
            
        eval_env.close()

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, all_frames, fps=30)
    print(f"Rollout GIF saved to {gif_path}")



if __name__ == "__main__":
    config = MAPPOConfig()

    # ideally can just say what the start conditions are:
    custom_start_conditions = {
        "custom_range": torch.tensor([0.2, 0.2]), 
        "custom_offset_blue": torch.tensor([0.0, 0.8]), 
        "spawn_in_formation": True,
    }
    
    device = setup_environment()
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    gif_path = f"./rollout_videos/{config.policy_folder_name}_checkpoint_{config.checkpoint_iteration}_{timestamp}_rollout.gif"
    
    temp_env = VmasEnv(
        scenario=config.scenario(), 
        num_envs=1, 
        max_steps=config.max_steps, 
        device=device,
        n_blue_agents=config.b_agents, 
        n_red_agents=config.r_agents,
        observe_teammates=config.observe_teammates,
    )
    agent_key = temp_env.action_keys[0][0]

    policy = build_mappo_modules(temp_env, config, device, agent_key)
    loaded_policy = load_policy_weights(policy, config, device)
    run_custom_rollout(loaded_policy, config, device, gif_path, custom_start_conditions)
    
    temp_env.close()