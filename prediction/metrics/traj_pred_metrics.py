# Copyright (c) Carizon. All rights reserved.
from typing import Optional

import torch
from metrics.traj_pred_metric_utils import topk, valid_filter
from torchmetrics import Metric


class minADE(Metric):

    def __init__(
        self,
        max_guesses: int = 6,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        super(minADE, self).__init__(**kwargs)
        self.add_state(
            "sum",
            default=torch.tensor(0.0, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor(0, device=device),
            dist_reduce_fx="sum",
        )
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        min_criterion: str = "FDE",
    ) -> None:

        pred, target, prob, valid_mask = valid_filter(
            pred,
            target,
            prob,
            valid_mask,
            keep_invalid_final_step,
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        if min_criterion == "FDE":
            inds_last = (
                valid_mask
                * torch.arange(
                    1, valid_mask.size(-1) + 1, device=valid_mask.device
                )
            ).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last]
                - target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                p=2,
                dim=-1,
            ).argmin(dim=-1)
            self.sum += (
                (
                    torch.norm(
                        pred_topk[torch.arange(pred.size(0)), inds_best]
                        - target,
                        p=2,
                        dim=-1,
                    )
                    * valid_mask
                ).sum(dim=-1)
                / valid_mask.sum(dim=-1)
            ).sum()
        elif min_criterion == "ADE":
            self.sum += (
                (
                    torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1)
                    * valid_mask.unsqueeze(1)
                )
                .sum(dim=-1)
                .min(dim=-1)[0]
                / valid_mask.sum(dim=-1)
            ).sum()
        else:
            raise ValueError(
                "{} is not a valid criterion".format(min_criterion)
            )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class minFDE(Metric):

    def __init__(
        self,
        max_guesses: int = 6,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        super(minFDE, self).__init__(**kwargs)
        self.add_state(
            "sum",
            default=torch.tensor(0.0, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor(0, device=device),
            dist_reduce_fx="sum",
        )
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
    ) -> None:
        pred, target, prob, valid_mask = valid_filter(
            pred,
            target,
            prob,
            valid_mask,
            keep_invalid_final_step,
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)
        inds_last = (
            valid_mask
            * torch.arange(
                1, valid_mask.size(-1) + 1, device=valid_mask.device
            )
        ).argmax(dim=-1)
        self.sum += (
            torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last]
                - target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                p=2,
                dim=-1,
            )
            .min(dim=-1)[0]
            .sum()
        )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


class Brier(Metric):

    def __init__(
        self,
        max_guesses: int = 6,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> None:
        super(Brier, self).__init__(**kwargs)
        self.add_state(
            "sum",
            default=torch.tensor(0.0, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.tensor(0, device=device),
            dist_reduce_fx="sum",
        )
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        min_criterion: str = "FDE",
    ) -> None:
        pred, target, prob, valid_mask = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, prob_topk = topk(self.max_guesses, pred, prob)
        if min_criterion == "FDE":
            inds_last = (
                valid_mask
                * torch.arange(
                    1, valid_mask.size(-1) + 1, device=valid_mask.device
                )
            ).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last]
                - target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                p=2,
                dim=-1,
            ).argmin(dim=-1)
        elif min_criterion == "ADE":
            inds_best = (
                (
                    torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1)
                    * valid_mask.unsqueeze(1)
                )
                .sum(dim=-1)
                .argmin(dim=-1)
            )
        else:
            raise ValueError(
                "{} is not a valid criterion".format(min_criterion)
            )
        self.sum += (
            (1.0 - prob_topk[torch.arange(pred.size(0)), inds_best])
            .pow(2)
            .sum()
        )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
