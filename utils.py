import warnings
from typing import Sequence
import torch
import argparse


SAVED_POLICIES = {
    "baseline": "./saved_policies/mappo_football_011225_195207/iteration_499_policy.pt",
    "mask_rhs": "./saved_policies/mappo_football_031225_133315/iteration_2950_policy.pt",
    "mask_lhs": "./saved_policies/mappo_football_091225_174907/iteration_1950_policy.pt",
    "mask_ths": "./saved_policies/mappo_football_311225_070808/iteration_950_policy.pt",
    "mask_bhs": "./saved_policies/mappo_football_311225_011123/iteration_1450_policy.pt",

    "mask_opp": "./saved_policies/mappo_football_301225_072508/iteration_950_policy.pt",
    "mask_bll": "./saved_policies/mappo_football_101225_181355/iteration_1450_policy.pt",
    "mask_opp_if_far": "./saved_policies/mappo_football_311225_072859/iteration_1950_policy.pt",
    "mask_ball_if_far": "./saved_policies/mappo_football_311225_011842/iteration_1450_policy.pt",
    "mask_opp_if_close": "./saved_policies/mappo_football_311225_071737/iteration_950_policy.pt",
    "mask_ball_if_close": "./saved_policies/mappo_football_311225_012232/iteration_1450_policy.pt",

    "baseline_v1": "./saved_policies/mappo_football_040226_200141/iteration_499_policy.pt",
    "baseline_v2": "./saved_policies/mappo_football_040226_200347/iteration_499_policy.pt"
}

class ClipModule(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val) 


def parse_args():
    """Argument parsing setup."""
    parser = argparse.ArgumentParser(description="MAPPO Football Training")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for RL football training")
    parser.add_argument("--timestamp", type=str, default=None, help="Shared timestamp for WandB grouping")
    return parser.parse_args()


def check_loss_values(advantage, loss_vals, subdata, agent_key):
    print("NON-FINITE loss detected")
    print("advantage min/max", advantage.min().item(), advantage.max().item())
    print("loss_objective:", loss_vals["loss_objective"])
    print("loss_critic:", loss_vals["loss_critic"])
    print("loss_entropy:", loss_vals["loss_entropy"])
    print("loc min/max:", subdata[(agent_key, "loc")].min().item(), subdata[(agent_key, "loc")].max().item())
    print("scale min/max:", subdata[(agent_key, "scale")].min().item(), subdata[(agent_key, "scale")].max().item())


def standardize(input, exclude_dims: Sequence[int] = (), mean=None, std=None):
    """
    Standardizes the input tensor with the possibility of excluding specific dims from the statistics.
    Useful when processing multi-agent data to keep the agent dimensions independent.

    Args:
        input (Tensor): the input tensor to be standardized.
        exclude_dims (Sequence[int]): dimensions to exclude from the statistics, can be negative. Default: ().
        mean (Tensor): a mean to be used for standardization. Must be of shape broadcastable to input. Default: None.
        std (Tensor): a standard deviation to be used for standardization. Must be of shape broadcastable to input. Default: None.

    """
    input_shape = input.shape
    exclude_dims = [
        d if d >= 0 else d + len(input_shape) for d in exclude_dims
    ]  # Make negative dims positive

    permuted_input, permutation = permute_excluded_dims(input, exclude_dims)
    normalized_shape_len = len(input_shape) - len(exclude_dims)

    # Like functional layer_norm (additionally takes mean and std)
    norm_perm_input = _standardize(
        permuted_input,
        normalized_shape=permuted_input.shape[-normalized_shape_len:],
        mean=mean,
        std=std,
    )

    # Reverse permutation
    inv_permutation = torch.argsort(torch.LongTensor(permutation)).tolist()
    norm_input = torch.permute(norm_perm_input, inv_permutation)
    return norm_input


def permute_excluded_dims(input, exclude_dims: Sequence[int] = ()):
    input_shape = input.shape
    exclude_dims = [
        d if d >= 0 else d + len(input_shape) for d in exclude_dims
    ]  # Make negative dims positive

    if len(set(exclude_dims)) != len(exclude_dims):
        raise ValueError("Exclude dims has repeating elements")
    if any(dim < 0 or dim >= len(input_shape) for dim in exclude_dims):
        raise ValueError(
            f"exclude_dims provided outside bounds for input of shape={input_shape}"
        )
    if len(exclude_dims) == len(input_shape):
        warnings.warn(
            "standardize called but all dims were excluded from the statistics, returning unprocessed input"
        )
        return input

    # Put all excluded dims in the beginning
    permutation = list(range(len(input_shape)))
    for dim in exclude_dims:
        permutation.insert(0, permutation.pop(permutation.index(dim)))
    permuted_input = input.permute(*permutation)
    return permuted_input, permutation


def _standardize(input, normalized_shape, mean=None, std=None):
    len_normalized = len(normalized_shape)
    if input.shape[-len_normalized:] != normalized_shape:
        raise ValueError(
            f"Normalized shape {normalized_shape} does not trailing input shape {input.shape[-len_normalized:]}"
        )
    if mean is None:
        mean = torch.mean(input, keepdim=True, dim=tuple(range(-len_normalized, 0)))
    if std is None:
        std = torch.std(input, keepdim=True, dim=tuple(range(-len_normalized, 0)))
    return (input - mean) / std.clamp_min(1e-6)
