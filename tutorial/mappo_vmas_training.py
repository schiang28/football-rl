import torch
import imageio, os
from tqdm import tqdm

from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.record.loggers.wandb import WandbLogger


class MAPPOConfig:
    scenario_name = "football"
    n_agents = 3
    max_steps = 100

    frames_per_batch = 6_000  # no. of team frames collected per training iteration
    n_iters = 5  # no. of sampling and training iterations

    num_epochs = 30  # no. of optimisation steps per training iteration
    minibatch_size = 400  # size of mini-batches in each optimisation step
    lr = 3e-4
    max_grad_norm = 1.0  # maximum norm for the gradients
    clip_epsilon = 0.2  # clip value for PPO loss
    gamma = 0.99
    lmbda = 0.9
    entropy_eps = 1e-4  # coefficient of the entropy term in PPO loss

    num_cells = 256
    depth = 2
    share_parameters_policy = True
    share_parameters_critic = True
    mappo = True


def setup_environment():
    """Determines device, sets seed and configures torch/tensordict settings."""
    torch.manual_seed(0)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    vmas_device = device
    set_composite_lp_aggregate(False).set()

    return vmas_device, device


def make_env(config, vmas_device, show_specs, show_keys):
    """Creates VMAS enviroment and applies transformations."""
    num_vmas_envs = config.frames_per_batch // config.max_steps

    env = VmasEnv(
        scenario=config.scenario_name,
        num_envs=num_vmas_envs,
        continuous_actions=True,
        max_steps=config.max_steps,
        device=vmas_device,
        # Scenario kwargs
        n_agents=config.n_agents,
    )

    if show_specs:
        print("action_spec:", env.full_action_spec)
        print("reward_spec:", env.full_reward_spec)
        print("done_spec:", env.full_done_spec)
        print("observation_spec:", env.observation_spec)
    if show_keys:
        print("action_keys:", env.action_keys)
        print("reward_keys:", env.reward_keys)
        print("done_keys:", env.done_keys)

    agent_key = "agent_blue"
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[(agent_key, "episode_reward")]),
    )
    check_env_specs(env)
    
    return env, agent_key


def build_mappo_modules(env, config, device, agent_key):
    """Creates and returns policy and critic modules based on config."""
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
        NormalParamExtractor(),
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

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[agent_key, "observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.n_agents,
        centralised=config.mappo,
        share_params=config.share_parameters_critic,
        device=device,
        depth=config.depth,
        num_cells=config.num_cells,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[(agent_key, "observation")],
        out_keys=[(agent_key, "state_value")],
    )

    return policy, critic


def train_mappo(config, env, policy, critic, agent_key, device, vmas_device, use_wandb):
    total_frames = config.frames_per_batch * config.n_iters

    collector = SyncDataCollector(
        env,
        policy,
        device=vmas_device,
        storing_device=device,
        frames_per_batch=config.frames_per_batch,
        total_frames=total_frames
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(config.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=config.minibatch_size,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_coeff=config.entropy_eps,
        normalize_advantage=False,
    )
    # tell the loss where to find the keys
    loss_module.set_keys( 
        reward=env.reward_key,
        action=env.action_key,
        value=(agent_key, "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=(agent_key, "done"),
        terminated=(agent_key, "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda
    )
    GAE = loss_module.value_estimator
    optim = torch.optim.Adam(loss_module.parameters(), config.lr)

    if use_wandb:
        logger = WandbLogger(
            exp_name=f"mappo_{config.scenario_name}",
            project="torchrl_mappo_vmas",
            log_dir="./wandb_logs"
        )
    else:
        class DummyLogger:
            def log_scalar(self, name, value, step): pass
        logger = DummyLogger()  

    pbar = tqdm(total=config.n_iters, desc="episode_reward_mean = 0")

    episode_reward_mean_list = [] # used for default terminal logger
    log_iteration = 0

    for tensordict_data in collector:
        tensordict_data.set(
            ("next", agent_key, "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", agent_key, "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )  # Compute GAE and add it to the data

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(config.num_epochs):
            for _ in range(config.frames_per_batch // config.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), config.max_grad_norm
                )

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        done = tensordict_data.get(("next", agent_key, "done"))
        if done.any():
            episode_rewards = tensordict_data.get(("next", agent_key, "episode_reward"))[done]
            episode_reward_mean = episode_rewards.mean().item()
            logger.log_scalar(name="episode_reward/mean", value=episode_reward_mean, step=log_iteration)
            pbar.set_description(f"episode_reward_mean = {episode_reward_mean:.2f}", refresh=False)
            

        pbar.update()
        log_iteration += 1

    return policy


def save_policy(policy, save_path):
    """Saves the state dictionary of trained policy."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"policy saved to {save_path}")


def record_rollout(policy, config, device, gif_path):
    """Runs a single episode rollout using policy and saves it as a GIF."""
    record_env = VmasEnv(
        scenario=config.scenario_name,
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



if __name__ == "__main__":
    config = MAPPOConfig()
    SAVE_POLICY = True
    SAVE_ROLLOUT = True

    policy_path = f"./saved_policies/mappo_{config.scenario_name}_policy.pt"
    gif_path = f"./rollout_videos/mappo_{config.scenario_name}_rollout.gif"

    vmas_device, device = setup_environment()
    env, agent_key = make_env(config, device, show_specs=False, show_keys=True)
    policy, critic = build_mappo_modules(env, config, device, agent_key)
    
    trained_policy = train_mappo(
        config=config,
        env=env,
        policy=policy,
        critic=critic,
        agent_key=agent_key,
        device=device,
        vmas_device=vmas_device,
        use_wandb=True
    )

    if SAVE_POLICY: save_policy(trained_policy, policy_path)
    if SAVE_ROLLOUT: record_rollout(trained_policy, config, device, gif_path)
