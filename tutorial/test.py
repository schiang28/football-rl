import torch
import imageio
import os
import datetime

from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

from torchrl.envs import TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from utils import ClipModule
from football_design import FootballDesign


class MAPPOConfig:
    # Environment
    max_steps = 500 # max no. timesteps per episode
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
    num_cells = 256 # size of each layer in nn
    depth = 2 # no. hidden layer in policy/value networks

    # Evaluation
    explore = False



def setup_environment():
    """Determines device, sets seed and configures torch/tensordict settings."""
    torch.manual_seed(0)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0 if torch.cuda.is_available() and not is_fork else "cpu")
    vmas_device = device
    set_composite_lp_aggregate(False).set()

    return vmas_device, device


def make_env(config, vmas_device, custom_blue_pos=None, custom_red_pos=None, custom_ball_pos=None):
    """Creates VMAS enviroment for simulation."""
    if config.scenario_name == "football": scenario = config.scenario()
    else: scenario = config.scenario_name

    env = VmasEnv(
        scenario=scenario,
        num_envs=1, # use num_envs=1 for recording a single rollout
        continuous_actions=True,
        max_steps=config.max_steps,
        device=vmas_device,
        n_blue_agents=config.b_agents,
        n_red_agents=config.r_agents,
        observe_teammates=config.observe_teammates,
        custom_blue_pos=custom_blue_pos,
        custom_red_pos=custom_red_pos,
        custom_ball_pos=custom_ball_pos
    )

    agent_key = env.action_keys[0][0]
    env = TransformedEnv(env)
    return env, agent_key


def build_mappo_modules(env, config, device, agent_key):
    """Creates policy module architecture to load weights into."""
    n_actions_per_agent = env.full_action_spec[env.action_key].shape[-1]
    n_obs_per_agent = env.observation_spec[agent_key, "observation"].shape[-1]

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=n_obs_per_agent,
            n_agent_outputs=2 * n_actions_per_agent,
            n_agents=env.n_agents,
            centralised=False,
            share_params=config.share_parameters_policy,
            device=device,
            depth=config.depth,
            num_cells=config.num_cells,
            activation_class=torch.nn.Tanh,
        ),
        ClipModule(-5, 5),
        NormalParamExtractor("biased_softplus_1.0"),
    )

    policy_module = TensorDictModule(
        policy_net, in_keys=[(agent_key, "observation")], out_keys=[(agent_key, "loc"), (agent_key, "scale")],
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

    # no critic is needed for simulation
    return policy, None


def get_custom_positions(start_pos_dict, device):
    blue_pos = start_pos_dict['blue_pos']
    red_pos = start_pos_dict['red_pos']
    ball_pos = start_pos_dict['ball_pos']

    if blue_pos:
        custom_blue_start = torch.tensor(
            blue_pos,
            device=device,
            dtype=torch.float32
        )
    else: custom_blue_start = None

    if red_pos:
        custom_red_start = torch.tensor(
            red_pos,
            device=device,
            dtype=torch.float32
        )
    else: custom_red_start = None

    if ball_pos:
        custom_ball_start = torch.tensor(
            ball_pos,
            device=device,
            dtype=torch.float32
        )
    else: custom_ball_start = None

    return custom_blue_start, custom_red_start, custom_ball_start


def simulate_rollout(checkpoint_path, config, gif_path, start_pos_dict):
    """Loads policy weights, runs a single episode, and saves the rollout as a GIF."""
    print(f"Starting simulation. Loading policy from: {checkpoint_path}")

    vmas_device, device = setup_environment()
    custom_blue_pos, custom_red_pos, custom_ball_pos = get_custom_positions(start_pos_dict, device)
    env, agent_key = make_env(config, vmas_device, custom_blue_pos, custom_red_pos, custom_ball_pos)
    policy, _ = build_mappo_modules(env, config, device, agent_key)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])

    all_frames = []

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = env.reset()
        dones = td.get("done", torch.zeros(1,1, dtype=torch.bool, device=device))

        for step in range(config.max_steps):
            if dones.all():
                print(f"Episode ended early at step {step}/{config.max_steps}.")
                break

            td = policy(td)
            td = env.step(td)
            frame = env.render(mode="rgb_array")
            all_frames.append(frame)
            dones = td.get("done", torch.zeros(1,1, dtype=torch.bool, device=device))

        env.close()

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, all_frames, fps=30)
    print(f"GIF saved to: {gif_path}")



if __name__ == "__main__":
    config = MAPPOConfig()

    experiment = "mappo_football_011225_195207"
    policy_number = "499"

    saved_checkpoint_path = f"./saved_policies/{experiment}/iteration_{policy_number}_policy.pt"
    timestamp = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    gif_path = f"./loaded_policy_rollouts/{experiment}_iter{policy_number}_{timestamp}.gif"

    start_pos_dic = {
        "blue_pos": [[-0.5, 0]],
        "red_pos": None,
        "ball_pos": None
    }

    simulate_rollout(saved_checkpoint_path, config, gif_path, start_pos_dic)