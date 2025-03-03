# Copyright (c) Carizon. All rights reserved.

import torch
import torch.distributions as D
import torch.nn.functional as F


class QCNetloss(torch.nn.Module):
    """QCNet loss."""

    def __init__(
        self,
    ):
        """Initialize method."""
        super(QCNetloss, self).__init__()

    def calc_nll(
        self,
        pred_locs: torch.Tensor,
        pred_scales: torch.Tensor,
        gt_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the negative log-likelyhood of the ground truth coordinates.

        Args:
            pred_locs: Predicted locs of the distributions. [B, K, N, T, 2]
            pred_scales: Predicted loss of the distributions. [B, K, N, T, 2]
            gt_coords: The ground truth coordinates.  [B, K, N, T, 2]

        Returns:
            nll: The negative log-likelyhood of the ground truth coordinates
                summed across all feature dimensions.  Shape [B, K, N, T]
        """
        pred_lap_coords = D.Laplace(pred_locs, pred_scales)
        nll = -pred_lap_coords.log_prob(gt_coords)
        return nll.sum(-1)  # sum across feature dimension

    def calc_mixture_nll(
        self,
        endpoint_pred_locs: torch.Tensor,
        endpoint_pred_scales: torch.Tensor,
        endpoint_targets: torch.Tensor,
        scores: torch.Tensor,
        cls_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate loss.

        Args:
            endpoint_pred_locs: The predicted locations of the endpoints.
                Shape [B, K, N, 2].
            endpoint_pred_scales: The predicted scales of the endpoints.
                Shape [B, K, N, 2].
            endpoint_targets: The ground truth coordinates of the endpoints.
                Shape [B, K, N, 2].
            scores: The predicted scores of the endpoints. Shape [B, K, N].
            cls_mask: The mask of the valid endpoints. Shape [B, N]

        Returns:
            mixture_nll: The endpoint mixture negative log-likelyhood.
                Shape [B, N]
        """
        nlls = self.calc_nll(
            pred_locs=endpoint_pred_locs,
            pred_scales=endpoint_pred_scales,
            gt_coords=endpoint_targets,
        )  # [B, K, N]
        nlls = nlls * cls_mask[:, None, :]  # [B, K, N]
        log_pis = F.log_softmax(scores, dim=1)  # [B, K, N]
        mixture_nll = -torch.logsumexp(log_pis - nlls, dim=1)
        return mixture_nll  # [B, N]

    def forward(self, data: dict):
        """
        Calculate losses.

        Args:
            data: A dict containing the input data. The following keys are
            needed:
            - agent_features.state_valid_masks: masks for future states.
            Shape [B, N, FT]
            - agent_features.acs/poses/fut: the ground truth future
            coordinates in
            agent-centric coordinate frames. Shape [B, N, FT, 3]
            - locs_stage1: the predicted locs of the first stage. Shape
            [B, K, N, FT, 2]
            - scales_stage1: the predicted scales of the first stage. Shape
            [B, K, N, FT, 2]
            - locs_stage2: the predicted locs of the second stage. Shape
            [B, K, N, FT, 2]
            - scales_stage2: the predicted scales of the second stage.
            Shape [B, K, N, FT, 2]
        Returns:
            loss_dict:
                "reg_propose_loss": first stage regression loss
                "reg_refine_loss": second stage regression loss
                "cls_loss": classification loss
                "total_loss": total_loss
        """
        reg_mask = data["agent_features"]["state_valid_masks"][
            "fut"
        ]  # [B, N, FT]
        cls_mask = reg_mask[:, :, -1]  # [B, N]
        locs_stage1 = data["locs_stage1"]  # [B, K, N, FT, 2]
        scales_stage1 = data["scales_stage1"]  # [B, K, N, FT, 2]
        locs_stage2 = data["locs_stage2"]  # [B, K, N, FT, 2]
        scales_stage2 = data["scales_stage2"]  # [B, K, N, FT, 2]
        pred_scores = data["scores"]  # [B, K, N]
        gt_coords = data["agent_features"]["acs"]["poses"]["fut"][
            ..., :2
        ]  # [B, N, FT, 2]
        ag_mask = reg_mask.any(-1)  # [B, N]
        B, K, N, FT, H = locs_stage1.shape

        # --- nll regression loss for the best Laplace component ---
        l2_norm = (
            torch.norm(
                locs_stage1 - gt_coords[:, None, :, :, :],
                dim=-1,
            )  # [B, K, N, FT]
            * reg_mask[:, None, :, :]
        ).sum(
            dim=-1
        )  # [B, K, N]
        best_mode = l2_norm.argmin(dim=1)  # [B, N]
        best_mode_index = best_mode[:, None, :, None, None].repeat(
            1, 1, 1, FT, H
        )
        locs_stage1_best = locs_stage1.gather(
            dim=1, index=best_mode_index
        ).squeeze(
            1
        )  # [B, N, FT, 2]
        scales_stage1_best = scales_stage1.gather(
            dim=1, index=best_mode_index
        ).squeeze(
            1
        )  # [B, N, FT, 2]
        best_component_nll_stage1 = self.calc_nll(
            pred_locs=locs_stage1_best,  # [B, N, FT, 2]
            pred_scales=scales_stage1_best,  # [B, N, FT, 2]
            gt_coords=gt_coords,  # [B, N, FT, 2]
        )  # [B, N, FT]
        best_component_nll_stage1 = best_component_nll_stage1 * reg_mask
        nll_loss_stage1 = (
            (
                best_component_nll_stage1.sum(-1)
                / (reg_mask.sum(-1).clamp_(min=1))
            ).sum(-1)
            / (ag_mask.sum(-1).clamp_(min=1))
        ).mean()

        locs_stage2_best = locs_stage2.gather(
            dim=1, index=best_mode_index
        ).squeeze(
            1
        )  # [B, N, FT, 3]
        scales_stage2_best = scales_stage2.gather(
            dim=1, index=best_mode_index
        ).squeeze(
            1
        )  # [B, N, FT, 3]
        best_component_nll_stage2 = self.calc_nll(
            pred_locs=locs_stage2_best,  # [B, N, FT, 3]
            pred_scales=scales_stage2_best,  # [B, N, FT, 3]
            gt_coords=gt_coords,  # [B, N, FT, 3]
        )  # [B, N, FT]
        best_component_nll_stage2 = best_component_nll_stage2 * reg_mask
        nll_loss_stage2 = (
            (
                best_component_nll_stage2.sum(-1)
                / (reg_mask.sum(-1).clamp_(min=1))
            ).sum(-1)
            / (ag_mask.sum(-1).clamp_(min=1))
        ).mean()

        # --- nll classification loss for scenes with endpoints considered ---
        cls_loss = self.calc_mixture_nll(
            endpoint_pred_locs=locs_stage2[
                :, :, :, -1, :
            ].detach(),  # [B, K, N, 2]
            endpoint_pred_scales=scales_stage2[
                :, :, :, -1, :
            ].detach(),  # [B, K, N, 2]
            endpoint_targets=gt_coords[:, None, :, -1, :],  # [B, 1, N, 2]
            scores=pred_scores,  # [B, K, N]
            cls_mask=cls_mask,  # [B, N]
        ).sum() / (B * cls_mask.sum())

        total_loss = nll_loss_stage1 + nll_loss_stage2 + cls_loss

        return {
            "reg_propose_loss": nll_loss_stage1,
            "reg_refine_loss": nll_loss_stage2,
            "cls_loss": cls_loss,
            "total_loss": total_loss,
        }
