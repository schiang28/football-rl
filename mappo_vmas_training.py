import torch
import os

from tqdm import tqdm
import datetime
import time
import numpy as np
from math import floor

from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.record.loggers.wandb import WandbLogger
import wandb

from football_design import FootballDesign
from utils import standardize, check_loss_values, ClipModule, SAVED_POLICIES, SAVED_POLICIES_2V1, parse_args
from logging_tools import DummyLogger
from custom_layers import GNNCommunicationLayer
from asymmetries import AsymmetryConfig, ObservationMasks, OpponentDifficulty

os.environ["PYGLET_HEADLESS"] = "true"


class MAPPOConfig:
    # Environment
    max_steps = 500 # max no. timesteps per episode (def. 500)
    scenario_name = "football"
    scenario = FootballDesign
    b_agents = 1
    r_agents = 1
    n_agents = b_agents + r_agents
    observe_teammates = False

    # Model
    mappo = True
    share_parameters_policy = b_agents == 1 # true if just 1v1 play, otherwise don't share params
    share_parameters_critic = b_agents == 1 # true if just 1v1 play, otherwse don't share params
    num_cells = 256 # size of each layer in nn
    depth = 2 # no. hidden layer in policy/value networks

    # Collector
    n_iters = 500 # no. of training iterations (def. 500)
    frames_per_batch = 240_000  # total timesteps across episodes in one batch per iteration (def 240,000)
    num_vmas_envs = frames_per_batch // max_steps

    # Loss
    normalize_advantage = True
    gamma = 0.99
    lmbda = 0.9
    entropy_eps = 0.005 # 0.005
    clip_epsilon = 0.2  # clip value for PPO loss

    # Training
    num_epochs = 45  # no. of optimisation steps per training iteration (def. 45)
    minibatch_size = 4096  # no. samples processed per optimisation step (def. 4096)
    lr = 5e-5 # learning rate (def. 5e-5)
    max_grad_norm = 5.0  # maximum norm for the gradients

    # Evaluation
    num_evaluations = 10
    evaluation_interval = n_iters // num_evaluations
    explore = False

    # Checkpoints
    num_checkpoints = 10 # how many policies will be saved during training
    checkpoint_interval = n_iters // num_checkpoints

    # Curriculum learning opponent
    max_speed = 0.15
    u_multiplier = 0.1



def setup_environment(seed):
    """Determines device, sets seed and configures torch/tensordict settings."""
    torch.manual_seed(seed)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0 if torch.cuda.is_available() and not is_fork else "cpu")
    vmas_device = device
    set_composite_lp_aggregate(False).set()

    return vmas_device, device


def make_env(config, vmas_device, asymmetries, show_specs, show_keys):
    """Creates VMAS enviroment and applies transformations."""
    if config.scenario_name == "football": scenario = config.scenario()
    else: scenario = config.scenario_name

    env = VmasEnv(
        scenario=scenario,
        num_envs=config.num_vmas_envs,
        continuous_actions=True,
        max_steps=config.max_steps,
        device=vmas_device,
        n_blue_agents=config.b_agents,
        n_red_agents=config.r_agents,
        observe_teammates=config.observe_teammates,
        **asymmetries
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

    agent_key = env.action_keys[0][0] # or "agents" or "agent_blue"
    env = TransformedEnv(env, RewardSum(in_keys=[env.reward_key], out_keys=[(agent_key, "episode_reward")]))
    check_env_specs(env)
    
    return env, agent_key


def build_mappo_modules(env, config, device, agent_key, use_gnn):
    """Creates and returns policy and critic modules based on config (MAPPO)."""
    n_actions_per_agent = env.full_action_spec[env.action_key].shape[-1]
    n_obs_per_agent = env.observation_spec[agent_key, "observation"].shape[-1]

    # setup policy network depending on whether to use a GNN to allow for communication when there is more than one agent
    if config.b_agents > 1 and use_gnn:
        policy_net = torch.nn.Sequential(
            GNNCommunicationLayer(n_obs_per_agent, config.num_cells, env.n_agents).to(device),
            MultiAgentMLP(
                n_agent_inputs=config.num_cells,
                n_agent_outputs=2 * n_actions_per_agent,
                n_agents=env.n_agents,
                centralised=False,
                share_params=config.share_parameters_policy,
                device=device,
                depth=config.depth - 1, # adjust since GNN is already one layer
                num_cells=config.num_cells,
                activation_class=torch.nn.Tanh,
            ),
            ClipModule(-5, 5),
            NormalParamExtractor("biased_softplus_1.0"),
        )
    else:
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

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec[agent_key, "observation"].shape[-1],
        n_agent_outputs=1,
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


def create_buffer(config, device):
    """Creates replay buffer."""
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(config.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=config.minibatch_size,
    )

    return replay_buffer


def create_collector(config, env, policy, vmas_device, device, total_frames):
    """Creates collector."""
    collector = SyncDataCollector(
        env,
        policy,
        device=vmas_device,
        storing_device=device,
        frames_per_batch=config.frames_per_batch,
        total_frames=total_frames
    )

    return collector


def create_loss(config, env, policy, critic, agent_key):
    """Creates loss modules."""
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=config.clip_epsilon,
        entropy_coeff=config.entropy_eps,
        normalize_advantage=False,
    )
    loss_module.set_keys( 
        reward=env.reward_key,
        action=env.action_key,
        value=(agent_key, "state_value"),
        done=(agent_key, "done"),
        terminated=(agent_key, "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=config.gamma, lmbda=config.lmbda)

    return loss_module


def setup_loggers(config, use_wandb, timestamp, local, seed, asymmetries):
    """Setup tqdm logger and WanDB logger if used."""
    group = asymmetries.label()
    group_name = f"{group}_{timestamp}"
    exp_name = f"{timestamp}_{group}_s{seed}"

    if use_wandb:
        project = "fb_mappo_tests" if local else "torchrl_mappo_vmas"
        logger = WandbLogger(exp_name=exp_name, project=project, log_dir="./wandb_logs", group=group_name)
    else: logger = DummyLogger()

    pbar = tqdm(total=config.n_iters, desc="episode_reward_mean = 0")

    return logger, pbar


def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", visualize_when_rgb=False))


def log_metrics(logger, pbar, done, current_tds, step, agent_key, training_tds, training_time, total_time, total_frames_collected, iteration_time, frames_per_batch):
    """Log relevant metrics in wandDB and to the terminal."""
    reward = current_tds.get(("next", agent_key, "reward"))

    # log training and learner metrics
    log_reward_dict = {
        "reward_mean": reward.mean().item(),
        "reward_max": reward.max().item(),
        "reward_min":reward.min().item(),
    }

    for key, val in log_reward_dict.items():
        logger.log_scalar(name=f"train/reward/{key}", value=val, step=step)
    logger.log_scalar(name="train/training_time", value=training_time, step=step)

    for key, val in training_tds.items():
        logger.log_scalar(name=f"train/learner/{key}", value=val.mean().item(), step=step)

    # log information metrics
    log_info_dict = {
        'total_time': total_time,
        'total_frames': total_frames_collected,
        'iteration_time': iteration_time,
        'current_frames': frames_per_batch
    }

    for key, val in log_info_dict.items():
        logger.log_scalar(name=f"info/{key}", value=val, step=step)
    logger.log_scalar(name="info/training_iteration", value=step, step=step)

    if done.any():
        episode_rewards = current_tds.get(("next", agent_key, "episode_reward"))[done]

        # compute reward stats
        log_episode_reward_dict = {
            "episode_reward_mean": episode_rewards.mean().item(),
            "episode_reward_max": episode_rewards.max().item(),
            "episode_reward_min": episode_rewards.min().item(),
        }

        for key, val in log_episode_reward_dict.items():
            logger.log_scalar(name=f"train/reward/{key}", value=val, step=step)

        # update to terminal progress bar
        pbar.set_description(f"episode_reward_mean = {log_episode_reward_dict['episode_reward_mean']:.2f}", refresh=False)

    pbar.update()


def log_evaluation_metrics(logger, rollouts, eval_env, evaluation_time, log_iteration, agent_key):
    """Log relevant evaluation metrics in WanDB including checkpointed videos."""
    rollouts = list(rollouts.unbind(0))
    for k, r in enumerate(rollouts):
        next_done = r.get(("next", "done")).sum(tuple(range(r.batch_dims, r.get(("next", "done")).ndim)), dtype=torch.bool)
        done_index = next_done.nonzero(as_tuple=True)[0][0]
        rollouts[k] = r[:done_index + 1]

    returns = [td.get(("next", agent_key, "reward")).sum(0).mean() for td in rollouts]
    rewards = [td.get(("next", agent_key, "reward")).mean() for td in rollouts]

    log_eval_dict = {
        "episode_reward_min": min(returns),
        "episode_reward_max": max(returns),
        "episode_reward_mean": sum(returns) / len(rollouts),
        "reward_mean": sum(rewards) / len(rollouts),
        "episode_len_mean": sum([td.batch_size[0] for td in rollouts]) / len(rollouts),
        "evaluation_time": evaluation_time,
    }

    video = torch.tensor(np.transpose(eval_env.frames[:rollouts[0].batch_size[0]], (0, 3, 1, 2)), dtype=torch.uint8).unsqueeze(0)

    for key, val in log_eval_dict.items():
        logger.log_scalar(name=f"eval/{key}", value=val, step=log_iteration)
    logger.experiment.log({f"eval/video": wandb.Video(video, fps=1 / eval_env.world.dt, format="mp4")}, commit=False)


def evaluate_agents(config, policy, logger, log_iteration, agent_key, device, asymmetries):
    """Evalute agents using current policy and log relevant evaluation metrics."""
    if config.scenario_name == "football": scenario = config.scenario()
    else: scenario = config.scenario_name

    eval_env = VmasEnv(
        scenario=scenario,
        num_envs=config.num_vmas_envs,
        continuous_actions=True,
        max_steps=config.max_steps,
        device=device,
        n_blue_agents=config.b_agents,
        n_red_agents=config.r_agents,
        observe_teammates=config.observe_teammates,
        render_mode="rgb_array",
        **asymmetries
    )

    evaluation_start_time = time.time()

    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM if config.explore else ExplorationType.DETERMINISTIC):
        eval_env.frames = []

        rollouts = eval_env.rollout(
            max_steps=config.max_steps,
            policy=policy,
            callback=rendering_callback,
            break_when_any_done=False,
        )

        evaluation_time = time.time() - evaluation_start_time
        log_evaluation_metrics(logger, rollouts, eval_env, evaluation_time, log_iteration, agent_key)


def get_opponent_strength(config, log_iteration, asymmetries, num_increments):
    """Use curriculum learning by setting AI opponent strength level by using a discrete number of increments to gradually step up AI strength ending at the defined maximum strength by end of training."""
    target_strength = asymmetries.opponent.ai_strength # when we want to go beyond additional difficulty for physical attributes
    if target_strength > 1.0: start_strength = 1.0
    else: start_strength = asymmetries.opponent.ai_decision_strength
    
    if num_increments == 1:
        return start_strength

    step_size = config.n_iters // num_increments
    curr_step_idx = min(floor(log_iteration / step_size), num_increments - 1)
    growth_per_step = (target_strength - start_strength) / (num_increments - 1)
    strength = start_strength + (curr_step_idx * growth_per_step)

    return strength


def get_checkpoints(config, policy, critic, optim, device, load_checkpoint_path, load_2v1_policy):
    """Load checkpoints into policy and critic, depending on if 1v1 or 2v1. Load individual policies into each first dimension of a multi-agent policy."""
    assert 1 <= config.b_agents <= 2, "number of agents for training not implemented. only 1 or 2 agents allowed."

    if config.b_agents == 1 or load_2v1_policy:
        # policy weight shape: [256, 24]
        checkpoint = torch.load(load_checkpoint_path[0], map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        log_iteration = checkpoint['iteration'] + 1
        total_frames_collected = log_iteration * config.frames_per_batch

    else: # if 2v1 scenario and 2 policies need to be loaded, loading each policy into a different branch of the network
        new_policy_dict = policy.state_dict()
        new_critic_dict = critic.state_dict()
        # new policy and critic weight shape: [2, 256, 24] and [2, 256, 48]

        for i in range(config.b_agents):
            checkpoint = torch.load(load_checkpoint_path[i], map_location=device)
            state_dict = checkpoint['policy_state_dict']
            critic_dict = checkpoint['critic_state_dict']
            # checkpoint policy and critic weights [256, 24]

            # copying policies into the correct dimensions of the network, critic is shared
            for key, value in state_dict.items():
                if key in new_policy_dict and isinstance(value, torch.Tensor):
                    new_policy_dict[key][i].copy_(value)
            
            if i == 0:
                for key, value in critic_dict.items():
                    if key in new_critic_dict and isinstance(value, torch.Tensor):
                        # the last dimension between new and loaded critic does not match so have to padd with 0s
                        if value.ndim == 2 and value.shape[-1] != new_critic_dict[key].shape[-1]:
                            last_dim = value.shape[-1]
                            new_critic_dict[key][0][:, :last_dim].copy_(value)
                            new_critic_dict[key][1][:, :last_dim].copy_(value)
                        else:
                            # for biases or hidden layers where dimensions match, might still have the [2, ...] agent dimension
                            if new_critic_dict[key].ndim > value.ndim:
                                new_critic_dict[key][0].copy_(value)
                                new_critic_dict[key][1].copy_(value)
                            else: new_critic_dict[key].copy_(value)

        policy.load_state_dict(new_policy_dict)
        critic.load_state_dict(new_critic_dict)
        log_iteration, total_frames_collected = 0, 0

    # to test returning log iteration and total frames colelcted as 0
    print(f"Loaded {len(load_checkpoint_path)} distinct policies into the team.")
    return log_iteration, total_frames_collected


def train_mappo(timestamp, seed, config, env, policy, critic, agent_key, device, vmas_device, use_wandb, save_policies, asymmetries, local, ai_increments, load_checkpoint_path, load_2v1_policy):
    """Main MAPPO algorithm training loop with evaluation and logging of metrics, returning the learnt policy for the agent."""
    total_frames = config.frames_per_batch * config.n_iters
    num_inner_iters = config.frames_per_batch // config.minibatch_size

    collector = create_collector(config, env, policy, vmas_device, device, total_frames)
    replay_buffer = create_buffer(config, device)
    loss_module = create_loss(config, env, policy, critic, agent_key)
    optim = torch.optim.Adam(loss_module.parameters(), config.lr)
    logger, pbar = setup_loggers(config, use_wandb, timestamp, local, seed, asymmetries)

    episode, log_iteration = 0, 0
    total_time = 0
    total_frames_collected = 0

    # If pre-trained policy provided, load that first
    if load_checkpoint_path: log_iteration, total_frames_collected = get_checkpoints(config, policy, critic, optim, device, load_checkpoint_path, load_2v1_policy)

    for tensordict_data in collector:
        iteration_start_time = time.time()

        # we need to expand the done and terminated to match the reward shape expected by the value estimator
        agent_next = ("next", agent_key)
        reward_shape = tensordict_data.get_item_shape(("next", env.reward_key))
        
        for key in ["done", "terminated"]:
            env_key = ("next", key)
            done_or_term = tensordict_data.get(env_key).unsqueeze(-1).expand(reward_shape)
            tensordict_data.set(agent_next + (key,), done_or_term)

        # value estimator and standardise excluding the agent dimension with gradient clipping
        with torch.no_grad():
            loss_module.value_estimator(tensordict_data, params=loss_module.critic_network_params, target_params=loss_module.target_critic_network_params)
            advantage = tensordict_data.get(loss_module.tensor_keys.advantage)

            if config.normalize_advantage and advantage.numel() > 1:
                advantage = standardize(advantage, exclude_dims=[-2])
                tensordict_data.set(loss_module.tensor_keys.advantage, advantage)

        # update replay buffer
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start_time = time.time()

        # optimisation loop
        for _ in range(config.num_epochs):
            for _ in range(num_inner_iters):
                subdata = replay_buffer.sample()

                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())
                loss_value = (loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"])
                if not torch.isfinite(loss_value): check_loss_values(advantage, loss_vals, subdata, agent_key)

                loss_value.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()
        iteration_end_time = time.time()
        training_tds = torch.stack(training_tds)

        # run evaluation every n episodes
        if (log_iteration > 0 and ((log_iteration % config.evaluation_interval == 0) or (episode + 1 == config.n_iters))):
            evaluate_agents(config, policy, logger, log_iteration, agent_key, device, asymmetries.to_env_kwargs())

        # save checkpointed policy every n episodes
        if (log_iteration > 0 and save_policies and ((log_iteration % config.checkpoint_interval == 0) or (episode + 1 == config.n_iters))):
            save_checkpoint(config, policy, log_iteration, critic, optim, timestamp, local, seed)

        training_time = iteration_end_time - training_start_time
        iteration_time = iteration_end_time - iteration_start_time
        total_time += iteration_time
        total_frames_collected += config.frames_per_batch

        done = tensordict_data.get(("next", agent_key, "done"))

        log_metrics(
            logger=logger,
            pbar=pbar,
            done=done,
            current_tds=tensordict_data,
            step=log_iteration,
            agent_key=agent_key,
            training_tds=training_tds,
            training_time=training_time,
            total_time=total_time,
            total_frames_collected=total_frames_collected,
            iteration_time=iteration_time,
            frames_per_batch=config.frames_per_batch
        )
        log_iteration += 1

        # set AI opponent strength if we are using more than one increment
        if ai_increments > 1 or asymmetries.opponent.ai_strength > 1:
            cl_strength = get_opponent_strength(config, episode, asymmetries, ai_increments)

            collector.env.scenario.ai_decision_strength = min(1.0, cl_strength)
            collector.env.scenario.ai_precision_strength = min(1.0, cl_strength)

            collector.env.scenario.max_speed = config.max_speed * cl_strength
            collector.env.scenario.u_multiplier = config.u_multiplier * cl_strength
        episode += 1

    return policy


def save_checkpoint(config, policy, checkpoint, critic, optim, timestamp, local, seed):
    """Saves the state dictionary of trained policy."""
    if local: checkpoint_path = f"./test_policies/mappo_{config.scenario_name}_{timestamp}/iter_{checkpoint}_s{seed}pol.pt"
    else: checkpoint_path = f"./saved_policies/mappo_{config.scenario_name}_{timestamp}/iter_{checkpoint}_s{seed}pol.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    state = {
        'iteration': checkpoint,
        'policy_state_dict': policy.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }

    torch.save(state, checkpoint_path)
    print(f"iteration {checkpoint} policy saved to {checkpoint_path}")



if __name__ == "__main__":
    args = parse_args()
    config = MAPPOConfig()
    LOAD_POLICY = True
    LOAD_2V1_POLICY = False
    SAVE_POLICY = False
    USE_WANDB = True
    LOCAL = True
    AI_INCREMENTS = 3
    GNN_COMMUNICATION = False

    asymmetries = AsymmetryConfig(
        masks=ObservationMasks(
            mask_pitch_rhs=[False, True],  # agent 0 sees rhs, agent 1 doesn't
        ),
        opponent=OpponentDifficulty(
            ai_strength=1.5,
            ai_decision_strength=1.0,
            ai_precision_strength=1.0,
        )
    )

    if LOCAL: # if running locally for testing
        config.n_iters = 100
        config.max_steps = 100
        config.frames_per_batch = 48000
        config.num_epochs = 10
        config.minibatch_size = 128

    if LOAD_POLICY:
        load_checkpoint_path = [SAVED_POLICIES["baseline"]] # first policy e.g. SAVED_POLICIES["baseline"]
        if config.b_agents > 1: load_checkpoint_path.append(SAVED_POLICIES["mask_rhs"]) # second policy if needed for multi-agent training
    else: load_checkpoint_path = None


    timestamp = args.timestamp if args.timestamp else datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    vmas_device, device = setup_environment(seed=args.seed)
    env, agent_key = make_env(config, device, asymmetries.to_env_kwargs(), show_specs=False, show_keys=True)
    policy, critic = build_mappo_modules(env, config, device, agent_key, use_gnn=GNN_COMMUNICATION)
    
    trained_policy = train_mappo(
        timestamp=timestamp,
        seed=args.seed,
        config=config,
        env=env,
        policy=policy,
        critic=critic,
        agent_key=agent_key,
        device=device,
        vmas_device=vmas_device,
        asymmetries=asymmetries,
        use_wandb=USE_WANDB,
        save_policies=SAVE_POLICY,
        local=LOCAL,
        ai_increments=AI_INCREMENTS,
        load_checkpoint_path=load_checkpoint_path,
        load_2v1_policy=LOAD_2V1_POLICY
    )