# Copyright (c) Carizon. All rights reserved.

import csv
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

# from horizon_plugin_pytorch.qtensor import QTensor


def take_row(in_tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """loop-free batched row-wise indexing.

    The behavior is equivalent to::

        res = torch.stack([t[i] for t, i in zip(in_tensor, index)], dim=0)

    and the result is::

        res[i, j] = in_tensor[i, index[i, j]]

    Args:
        in_tensor (torch.Tensor): Input tensor with shape (B, N, ...).
        index (torch.Tensor): Index tensor with shape (B, M), where
            each entry should be less than N.
    """
    arr = torch.arange(in_tensor.shape[0], device=index.device)[:, None]
    flatten_index = (index + arr * in_tensor.shape[1]).flatten()

    last_dims = in_tensor.shape[2:]
    flatten_target = in_tensor.view(-1, *last_dims)
    if flatten_target.shape[0] == 0:
        flatten_target = (
            torch.unsqueeze(torch.ones(last_dims, device=index.device), dim=0)
            * -1
        )
    indexed = flatten_target[flatten_index.type(torch.long)].view(
        in_tensor.shape[0], -1, *last_dims
    )
    return indexed


def insert_row(
    in_tensor: torch.Tensor,
    index: torch.Tensor,
    target: Union[int, float, torch.Tensor],
) -> None:
    """Insert target to in_tensor by index provide by index param.

    The behavior is equivalent to::

      torch.stack([t[i] for t, i in zip(in_tensor, index)], dim=0) = target[i]

    and the result is::

      in_tensor[i, index[i, j]] = target[i, j]

    while the in_tensor will be modified by target.

    Args:
        in_tensor (torch.Tensor): Input tensor with shape (B, N, ...).
        index (torch.Tensor): Index tensor with shape (B, M), where
            each entry should be less than N.
        target (int, float, torch.Tensor): Target tensor provided.
            If target is torch.Tensor, it must be with shape (B, M).
    """

    last_dims = in_tensor.shape[2:]

    arr = torch.arange(in_tensor.shape[0], device=index.device)[:, None]
    flatten_index = (index + arr * in_tensor.shape[1]).flatten().long()

    flatten_target = in_tensor.view(-1, *last_dims)
    if isinstance(target, torch.Tensor):
        target = target.view(flatten_target[flatten_index].shape)
    flatten_target[flatten_index] = target


def select_sample(data: Any, bool_index: Union[List, torch.Tensor]):
    r"""Select sample according to bool index, return a tensor after selecting.

    Args:
        data : torch.tensor/QTensor/a list of tensor/ a dict of tensor,
            each tensor`s shape is (b,...)
        bool_index : torch.tensor, shape is (b,)

    """

    if isinstance(data, torch.Tensor):
        return data[bool_index]

    # elif type(data) == QTensor:
    #     return QTensor(data[bool_index], data.scale.clone(), data.dtype)

    elif isinstance(data, (list, tuple)):
        return [select_sample(x, bool_index) for x in data]

    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = select_sample(v, bool_index)
        return res
    else:
        raise TypeError("donot support the type to select")


def mean_with_mask(
    x: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Mean of elements along the last dimension in x with True flags.

    Args:
        x: Input tensor or ndarray with any shapes.
        mask: Mask, which can be broadcast to x.
    """
    if isinstance(x, torch.Tensor):
        agent_num = torch.sum(mask, dim=-1)
        x = torch.sum(x * mask, dim=-1) / torch.clamp(agent_num, min=1.0)
    elif isinstance(x, np.ndarray):
        agent_num = np.sum(mask, axis=-1)
        x = np.sum(x * mask, axis=-1) / np.clip(
            agent_num, a_max=None, a_min=1.0
        )
    else:
        raise NotImplementedError(f"unspport input type {type(x)}")
    return x


def divide_no_nan(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Divide torch Tensors safely."""
    div = x / y
    return torch.nan_to_num(div, nan=0, posinf=0, neginf=0)


def tensor_mean(tensor_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Compute mean value from a list of scalar tensors."""
    return torch.stack(tensor_list).mean()


def recursive_to_device(
    data: Union[Dict[str, torch.tensor], List[torch.tensor], torch.tensor],
    device: str,
):
    """
    Recursively move all tensors in a nested dictionary to the specified
    device.
    """
    if isinstance(data, dict):
        return {
            key: recursive_to_device(value, device)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [recursive_to_device(item, device) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def create_directories_and_file(
    base_path: str, task_name: str
) -> Tuple[str, str, str, str]:
    """
    Create necessary directories and a CSV file for logging training metrics.

    This function creates directories for checkpoints, visualizations, and
    TensorBoard logs, as well as initializes a CSV file for recording training
    metrics if it does not already exist.

    Args:
        base_path (str): The base directory where the task-specific directories
            will be created.
        task_name (str): The name of the task, used to create subdirectories
            for the task.

    Returns:
        Tuple[str, str, str, str]: Paths to the checkpoint directory,
                                   visualization directory,
                                   TensorBoard directory,
                                   and the metrics CSV file.
    """
    ckpt_dir = os.path.join(base_path, task_name, "checkpoint")
    viz_dir = os.path.join(base_path, task_name, "viz")
    tensorboard_dir = os.path.join(base_path, task_name, "tensorboard")
    file_path = os.path.join(ckpt_dir, "training_metrics.csv")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    if not os.path.exists(file_path):
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Epoch",
                    "Loss",
                    "FDE(K=6)",
                    "FDE(K=1)",
                    "ADE(K=6)",
                    "ADE(K=1)",
                    "MR(K=6)",
                    "MR(K=1)",
                    "Brier",
                    "LaneMissRate(K=1)",
                ]
            )

    return ckpt_dir, viz_dir, tensorboard_dir, file_path


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of parameters of a pytorch model.

    Args:
        model(torch.nn.Module): a pytorch model that you want to count
        the number of parameters.

    Returns:
        num_params(int): The total number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
