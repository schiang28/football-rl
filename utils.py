import warnings
from typing import Sequence
import torch


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
