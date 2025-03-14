# Copyright (c) Carizon. All rights reserved.

from typing import Optional, Union

import torch


def sigmoid_and_clip(
    x: torch.tensor, min_value: float = 1e-4, max_value: float = 1 - 1e-4
) -> torch.tensor:
    """
    Apply the sigmoid function to the input tensor and clip the output
    to avoid Nan outputs.

    Args:
        x (torch.Tensor): The input tensor to which the sigmoid function is
            applied.
        min_value (float, optional): The minimum value to clip the output.
            Default is 1e-4.
        max_value (float, optional): The maximum value to clip the output.
            Default is 1 - 1e-4.

    Returns:
        torch.Tensor: The tensor after applying the sigmoid function and
            clipping.
    """
    y = torch.clamp(x.sigmoid(), min=min_value, max=max_value)
    return y


def weight_reduce_loss(
    loss: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: Union[str, callable] = "mean",
    avg_factor: Optional[float] = None,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Weight loss and then reduce.

    Args:
        loss: Losses.
        weight: Loss weight.
        reduction: The way to reduce loss.
        Function reduction is called in the following way:

        .. code-block:: python

            loss = reduction(loss)

        avg_factor: Avg factor of loss.
        inplace: Whether weighting loss inplace.
    Returns:
        return losses.
    """

    if weight is not None:
        if inplace:
            loss *= weight
        else:
            loss = loss * weight

    if avg_factor is None:
        if callable(reduction):
            return reduction(loss)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def gather_feat(
    feat: torch.Tensor, ind: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """Gather features according to indices and mask.

    Args:
        feat (torch.Tensor): 3-D feature tensor. The 1st dimension should be
            the batch dimension, and the last dimension should be the channel
            dimension.
        ind (torch.Tensor): index tensor. Should be 2-D. Will be extended to
            3-D in the code.
        mask (torch.Tensor, optional): post gathering mask. Defaults to None.

    Returns:
        [torch.Tensor]: a new tensor whose values are extracted from feat.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(
    feat: torch.Tensor, ind: torch.Tensor
) -> torch.Tensor:
    """Transpose features, then gather features according to indices.

    Args:
        feat (torch.Tensor): 4-D feature tensor of shape BxCxHxW.
        ind (torch.Tensor): 2-D index tensor.

    Returns:
        [torch.Tensor]: a new tensor whose values are extracted from feat.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat
