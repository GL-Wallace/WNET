# Copyright (c) Carizon. All rights reserved.

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_modules import FourierEmbedding, MLPLayer, RPEMHALayer
from utils import angle_between_2d_vectors, wrap_angle


class QCNetHead(nn.Module):
    """QCNet Decoder."""

    def __init__(
        self,
        num_historical_steps: int,
        num_future_steps: int,
        num_recurrent_steps: int,
        num_modes: int,
        max_agents: int,
        num_neighbors_a2pl: int,
        num_neighbors_a2a: int,
        hidden_dim: int,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ):
        """Initialize.

        Args:
            num_historical_steps (int): The number of historical steps.
            num_future_steps (int): The number of future steps.
            num_recurrent_steps (int): The number of recurrent steps.
            num_modes (int): The number of modes.
            max_agents (int): The maximum number of agents.
            num_neighbors_a2pl (int): The number of neighbors for agent to
                polyline.
            num_neighbors_a2a (int): The number of neighbors for agent to
                agent.
            hidden_dim (int): The hidden dimension.
            num_freq_bands (int): The number of frequency bands.
            num_layers (int): The number of layers.
            num_heads (int): The number of heads.
            head_dim (int): The head dimension.
            dropout (float): The dropout rate.
        """
        super(QCNetHead, self).__init__()
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_recurrent_steps = num_recurrent_steps
        self.num_modes = num_modes
        self.max_agents = max_agents
        self.num_neighbors_a2pl = num_neighbors_a2pl
        self.num_neighbors_a2a = num_neighbors_a2a
        self.hidden_dim = hidden_dim
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        self.build_nets()

    def build_nets(self):
        self.emb_dim_rpe_m2t = 4
        self.emb_dim_rpe_m2pl = 3
        self.emb_dim_rpe_m2a = 3

        self.mode_queries = nn.Parameter(
            data=torch.randn(
                self.num_modes,
                self.hidden_dim,
            )
        )
        self.emb_rpe_m2t = FourierEmbedding(
            input_dim=self.emb_dim_rpe_m2t,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.emb_rpe_m2pl = FourierEmbedding(
            input_dim=self.emb_dim_rpe_m2pl,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.emb_rpe_a2a = FourierEmbedding(
            input_dim=self.emb_dim_rpe_m2a,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )

        self.net_mode2sce_stage1 = ModeToSceneBlock(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.net_m2m_stage1 = RPEMHALayer(
            d_input=self.hidden_dim,
            d_model=self.head_dim,
            n_head=self.num_heads,
            dropout=self.dropout,
        )
        recur_time_steps = self.num_future_steps // self.num_recurrent_steps
        self.net_locs_stage1 = MLPLayer(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=recur_time_steps * 2,
        )
        self.net_scales_stage1 = MLPLayer(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=recur_time_steps * 2,
        )

        self.traj_emb_fourier = FourierEmbedding(
            input_dim=2,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.traj_emb_gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
        )
        self.net_mode2sce_stage2 = ModeToSceneBlock(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.net_m2m_stage2 = RPEMHALayer(
            d_input=self.hidden_dim,
            d_model=self.head_dim,
            n_head=self.num_heads,
            dropout=self.dropout,
        )
        self.net_locs_stage2 = MLPLayer(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_future_steps * 2,
        )
        self.net_scales_stage2 = MLPLayer(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.num_future_steps * 2,
        )

        self.net_mode_scores = MLPLayer(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=1,
        )

    def build_queries(
        self,
        mode_queries: torch.Tensor,
        scene_enc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Build initial queries for the module.

        Args:
            mode_queries: the learnable mode queries.
            scene_enc: the encoder output.

        Returns:
            {
            "x_m": the mode queries, [B, K, N, H]
            "x_a": the agent queries, [B*K*N, HT, H]
            "x_pl": the polyline queries, [B*K*N, M, H]
            }
        """
        B, N, HT, _ = scene_enc["x_a"].shape
        K = self.num_modes
        M = scene_enc["x_pl"].shape[1]

        x_m = mode_queries[None, :, None, :].repeat(B, 1, N, 1)  # [B, K, N, H]
        x_a = (
            scene_enc["x_a"][:, None, :, :, :]
            .repeat(1, K, 1, 1, 1)
            .reshape(B * K * N, HT, -1)
        )  # [B*K*N, HT, H]
        x_pl = (
            scene_enc["x_pl"][:, None, None, :, :]
            .repeat(1, K, N, 1, 1)
            .reshape(B * K * N, M, -1)
        )  # [B*K*N, M, H]

        return {
            "x_m": x_m,  # [B, K, N, H]
            "x_a": x_a,  # [B*K*N, HT, H]
            "x_pl": x_pl,  # [B*K*N, M, H]
        }

    def build_rpes(
        self,
        batch: dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Build the relative positional encodings for the module.

        Args:
            batch: input data dict.

        Returns:
            "rpe_m2t": rpe for mode to time, [B*K*N, 1, HT, H]
            "rpe_m2pl": rpe for mode to polyline, [B*K*N, 1, M, H]
            "rpe_m2a": rpe for mode to agents, [B*K, N, N, H]
            "knn_idxs_m2pl": knn indices for mode to polyline,
            [B*K*N, 1, NB_m2pl]
            "knn_idxs_m2a": knn indices for mode to agent, [B*K, N, NB_m2a]
        """
        agent_poses = batch["agent_features"]["gcs"]["poses"][
            "his"
        ]  # [B, N, HT, 3]
        agent_pos = agent_poses[..., :2]  # [B, N, HT, 2]
        agent_h = agent_poses[..., 2]  # [B, N, HT]
        mode_pos = agent_pos[:, :, -1, :]  # [B, N, 2]
        mode_h = agent_h[:, :, -1]  # [B, N]
        pl_poses = batch["map_features"]["gcs"]["pl_poses"]  # [B, M, 3]
        pl_pos = pl_poses[..., :2]  # [B, M, 2]
        pl_h = pl_poses[..., 2]  # [B, M]

        B, N, HT, _ = agent_poses.shape
        K = self.num_modes
        M = pl_poses.shape[1]

        mode_h_vec = torch.stack(
            [mode_h.cos(), mode_h.sin()], dim=-1
        )  # [B, N, 2]

        # mode to hist rpe
        rel_pos_m2h = agent_pos - mode_pos[:, :, None, :]  # [B, N, HT, 2]
        rel_h_m2h = wrap_angle(agent_h - mode_h[:, :, None])  # [B, N, HT]
        rpe_m2t = torch.stack(
            [
                torch.norm(rel_pos_m2h, dim=-1),  # [B, N, HT]
                angle_between_2d_vectors(
                    ctr_vector=mode_h_vec[:, :, None, :],  # [B, N, 1, 2]
                    nbr_vector=rel_pos_m2h,  # [B, N, HT, 2]
                ),  # [B, N, HT]
                rel_h_m2h,  # [B, N, HT]
                torch.arange(-HT + 1, 1, device=mode_pos.device)[
                    None, None, :
                ].repeat(
                    B, N, 1
                ),  # [B, N, HT]
            ],
            dim=-1,
        )  # [B, N, HT, 4]
        rpe_m2t = self.emb_rpe_m2t(
            continuous_inputs=rpe_m2t,
            categorical_embs=None,
        )  # [B, N, HT, H]
        rpe_m2t = (
            rpe_m2t[:, None, :, :, :]
            .repeat(1, K, 1, 1, 1)
            .reshape(B * K * N, 1, HT, -1)
        )  # [B*K*N, 1, T, H]

        # mode to polyline rpe
        rel_pos_m2pl = (
            pl_pos[:, None, :, :] - mode_pos[:, :, None, :]
        )  # [B, N, M, 2]
        rel_h_m2pl = wrap_angle(
            pl_h[:, None, :] - mode_h[:, :, None]
        )  # [B, N, M]
        dist_m2pl = torch.norm(rel_pos_m2pl, dim=-1)  # [B, N, M]
        knn_idxs_m2pl = torch.topk(
            -dist_m2pl, self.num_neighbors_a2pl, dim=-1
        ).indices  # [B, N, NB_m2pl]
        knn_idxs_m2pl = (
            knn_idxs_m2pl[:, None, :, :]
            .repeat(1, K, 1, 1)
            .reshape(B * K * N, 1, -1)
        )  # [B*K*N, 1, NB_m2pl]

        rpe_m2pl = torch.stack(
            [
                dist_m2pl,  # [B, N, M]
                angle_between_2d_vectors(
                    ctr_vector=mode_h_vec[:, :, None, :],  # [B, N, 1, 2]
                    nbr_vector=rel_pos_m2pl,  # [B, N, M, 2]
                ),  # [B, N, M]
                rel_h_m2pl,  # [B, N, M]
            ],
            dim=-1,
        )  # [B, N, M, 4]
        rpe_m2pl = self.emb_rpe_m2pl(
            continuous_inputs=rpe_m2pl,
            categorical_embs=None,
        )  # [B, N, M, H]
        rpe_m2pl = (
            rpe_m2pl[:, None, :, :, :]
            .repeat(1, K, 1, 1, 1)
            .reshape(B * K * N, 1, M, -1)
        )  # [B*K*N, 1, M, H]

        # mode to agent rpe
        rel_pos_m2a = (
            mode_pos[:, :, None, :] - mode_pos[:, None, :, :]
        )  # [B, N, N, 2]
        rel_h_m2a = wrap_angle(
            mode_h[:, :, None] - mode_h[:, None, :]
        )  # [B, N, N]
        dist_m2a = torch.norm(rel_pos_m2a, dim=-1)  # [B, N, N]
        knn_idxs_m2a = torch.topk(
            -dist_m2a, self.num_neighbors_a2a, dim=-1
        ).indices  # [B, N, NB_m2a]
        knn_idxs_m2a = (
            knn_idxs_m2a[:, None, :, :]
            .repeat(1, K, 1, 1)
            .reshape(B * K, N, -1)
        )  # [B*K, N, NB_m2a]

        rpe_m2a = torch.stack(
            [
                dist_m2a,  # [B, N, N]
                angle_between_2d_vectors(
                    ctr_vector=mode_h_vec[:, :, None, :],  # [B, N, 1, 2]
                    nbr_vector=rel_pos_m2a,  # [B, N, N, 2]
                ),  # [B, N, N]
                rel_h_m2a,  # [B, N, N]
            ],
            dim=-1,
        )  # [B, N, N, 3]
        rpe_m2a = self.emb_rpe_a2a(
            continuous_inputs=rpe_m2a,
            categorical_embs=None,
        )  # [B, N, N, H]
        rpe_m2a = (
            rpe_m2a[:, None, :, :, :]
            .repeat(1, K, 1, 1, 1)
            .reshape(B * K, N, N, -1)
        )  # [B*K, N, N, H]

        return {
            "rpe_m2t": rpe_m2t,  # [B*K*N, 1, HT, H]
            "rpe_m2pl": rpe_m2pl,  # [B*K*N, 1, M, H]
            "rpe_m2a": rpe_m2a,  # [B*K, N, N, H]
            "knn_idxs_m2pl": knn_idxs_m2pl,  # [B*K*N, 1, NB_m2pl]
            "knn_idxs_m2a": knn_idxs_m2a,  # [B*K, N, NB_m2a]
        }

    def build_masks(
        self,
        batch: dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Build the invalid masks for attention.

        Args:
            batch: input data dict.

        Returns:
                "invalid_mask_m2t": invalid mask for mode to time,
                [B*K*N, 1, HT]
                "invalid_mask_m2pl": invalid mask for mode to polyline,
                [B*K*N, 1, M]
                "invalid_mask_m2a": inlvalid mask for mode to agent,
                [B*K, N, N]
        """
        agent_state_valid_mask = batch["agent_features"]["state_valid_masks"][
            "his"
        ]  # [B, N, HT]
        agent_valid_mask = batch["agent_features"][
            "agent_valid_masks"
        ]  # [B, N]
        pl_valid_mask = batch["map_features"]["pl_valid_masks"]  # [B, M]

        B, N, HT = agent_state_valid_mask.shape
        M = pl_valid_mask.shape[1]
        K = self.num_modes

        invalid_mask_m2t = ~(
            agent_state_valid_mask[:, None, :, :].repeat(1, K, 1, 1)
        ).reshape(
            B * K * N, 1, HT
        )  # [B*K*N, 1, HT]
        invalid_mask_m2pl = ~(
            pl_valid_mask[:, None, None, :]
            .repeat(1, K, N, 1)
            .reshape(B * K * N, 1, M)
        )  # [B*K*N, 1, M]
        invalid_mask_m2a = ~(
            torch.logical_and(
                agent_valid_mask[:, None, :],
                agent_valid_mask[:, :, None],
            )[:, None, :, :]
            .repeat(1, K, 1, 1)
            .reshape(B * K, N, N)
        )  # [B*K, N, N]

        return {
            "invalid_mask_m2t": invalid_mask_m2t,  # [B*K*N, 1, HT]
            "invalid_mask_m2pl": invalid_mask_m2pl,  # [B*K*N, 1, M]
            "invalid_mask_m2a": invalid_mask_m2a,  # [B*K, N, N]
        }

    def forward(
        self,
        data: dict,
    ) -> Dict[str, torch.torch.Tensor]:
        """
        Forward pass of the module.

        Args:
            data: input data dict.

        Returns:
            "locs_stage1": predicted locs from stage1, [B, K, N, FT, 2]
            "scales_stage1": predicted scales from stage1, [B, K, N, FT, 2]
            "locs_stage2": predicted locs from stage2, [B, K, N, FT, 2]
            "scales_stage2": predicted scales from stage2, [B, K, N, FT, 2]
            "scores": mode scores, [B, K, N]
        """
        his_agent_poses = data["agent_features"]["gcs"]["poses"][
            "his"
        ]  # [B, N, HT, 3]
        (B, N, FT, K) = (
            his_agent_poses.size(0),
            his_agent_poses.size(1),
            self.num_future_steps,
            self.num_modes,
        )

        queries = self.build_queries(self.mode_queries, data["scene_enc"])
        x_m, x_a, x_pl = (queries["x_m"], queries["x_a"], queries["x_pl"])
        rpes = self.build_rpes(data)
        invalid_masks = self.build_masks(data)

        # --- first stage anchor proposals ---
        locs_stage1 = [None] * self.num_recurrent_steps
        scales_stage1 = [None] * self.num_recurrent_steps
        for t in range(self.num_recurrent_steps):
            x_m = self.net_mode2sce_stage1(
                x_m=x_m,
                x_a=x_a,
                x_pl=x_pl,
                rpes=rpes,
                invalid_masks=invalid_masks,
            )  # [B, K, N, H]

            x_m = x_m.transpose(1, 2).reshape(B * N, K, -1)  # [B*N, K, H]
            x_m = self.net_m2m_stage1(q=x_m, k=x_m, v=x_m)
            x_m = x_m.reshape(B, N, K, -1).transpose(1, 2)  # [B, K, N, H]

            locs_stage1[t] = self.net_locs_stage1(x_m)  # [B, K, N, FT*2//RT]
            scales_stage1[t] = self.net_scales_stage1(
                x_m
            )  # [B, K, N, FT*2//RT]

        locs_stage1 = torch.cat(locs_stage1, dim=-1).reshape(
            B, K, N, FT, -1
        )  # [B, K, N, FT, 2]
        locs_stage1[..., -1] = torch.tanh(locs_stage1[..., -1]) * math.pi
        locs_stage1 = torch.cumsum(locs_stage1, dim=-2)  # [B, K, N, FT, 2]
        scales_stage1 = torch.cat(scales_stage1, dim=-1).reshape(
            B, K, N, FT, -1
        )  # [B, K, N, FT, 3]
        scales_stage1 = (
            F.elu_(scales_stage1, alpha=1.0) + 1.0
        )  # [B, K, N, FT, 3]
        scales_stage1 = torch.cumsum(scales_stage1, dim=-2)
        scales_stage1.clamp_(min=0.1)

        # --- second stage refined trajs ---
        x_m = (
            self.traj_emb_fourier(locs_stage1.detach())
            .reshape(B * K * N, FT, -1)
            .permute(1, 0, 2)
        )  # [FT, B*K*N H]
        x_m = (
            self.traj_emb_gru(x_m)[1].squeeze(0).reshape(B, K, N, -1)
        )  # [B, K, N, H]
        x_m = self.net_mode2sce_stage2(
            x_m=x_m,
            x_a=x_a,
            x_pl=x_pl,
            rpes=rpes,
            invalid_masks=invalid_masks,
        )  # [B, K, N, H]

        x_m = x_m.transpose(1, 2).reshape(B * N, K, -1)  # [B*N, K, H]
        x_m = self.net_m2m_stage2(q=x_m, k=x_m, v=x_m)  # [B*N, K, H]
        x_m = x_m.reshape(B, N, K, -1).transpose(1, 2)  # [B, K, N, H]

        locs_stage2 = self.net_locs_stage2(x_m).reshape(
            B, K, N, FT, -1
        )  # [B, K, N, FT, 2]
        locs_stage2 = locs_stage2 + locs_stage1.detach()
        scales_stage2 = (
            F.elu_(self.net_scales_stage2(x_m), alpha=1.0) + 1.0
        ).reshape(
            B, K, N, FT, -1
        )  # [B, K, N, FT, 2]
        scales_stage2.clamp_(min=0.1)

        # --- caculate mode scores ---
        scores = self.net_mode_scores(x_m).squeeze(-1)  # [B, K, N]
        # TO-DO:

        return {
            "locs_stage1": locs_stage1,  # [B, K, N, FT, 2]
            "scales_stage1": scales_stage1,  # [B, K, N, FT, 2]
            "locs_stage2": locs_stage2,  # [B, K, N, FT, 2]
            "scales_stage2": scales_stage2,  # [B, K, N, FT, 2]
            "scores": scores,  # [B, K, N]
        }


class ModeToSceneBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        head_dim: int,
        num_heads: int,
        dropout: float,
    ):
        """Initialize.

        Args:
            num_layers (int): The number of layers.
            hidden_dim (int): The hidden dimension.
            head_dim (int): The head dimension.
            num_heads (int): The number of heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.net_m2t = nn.ModuleList(
            [
                RPEMHALayer(
                    d_input=self.hidden_dim,
                    d_model=self.head_dim,
                    n_head=self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.net_m2pl = nn.ModuleList(
            [
                RPEMHALayer(
                    d_input=self.hidden_dim,
                    d_model=self.head_dim,
                    n_head=self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.net_m2a = nn.ModuleList(
            [
                RPEMHALayer(
                    d_input=self.hidden_dim,
                    d_model=self.head_dim,
                    n_head=self.num_heads,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self,
        x_m: torch.Tensor,
        x_a: torch.Tensor,
        x_pl: torch.Tensor,
        rpes: Dict[str, torch.Tensor],
        invalid_masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward.

        Args:
            x_m (tensor): the mode queries, [B, K, N, H]
            x_a (tensor): the agent queries, [B*K*N, HT, H]
            x_pl (tensor): the polyline queries, [B*K*N, M, H]
            rpes (dict): the relative positional encodings.
            invalid_masks (dict): the invalid masks.

        Returns:
            x_m (tensor): the mode queries, [B, K, N, H]
        """
        B, K, N, _ = x_m.shape
        cur_x_a = x_a[:, -1, :].reshape(B * K, N, -1)
        for layer_id in range(self.num_layers):
            # mode to time cross attention
            x_m = x_m[:, :, :, None, :].reshape(B * K * N, 1, -1)
            x_m = self.net_m2t[layer_id](
                q=x_m,  # [B*K*N, 1, H]
                k=x_a,  # [B*K*N, T, H]
                v=x_a,  # [B*K*N, T, H]
                rpe=rpes["rpe_m2t"],  # [B*K*N, 1, T, H]
                invalid_mask=invalid_masks[
                    "invalid_mask_m2t"
                ],  # [B*K*N, 1, T]
            ).reshape(
                B, K, N, -1
            )  # [B, K, N, H]

            # mode to polyline cross attention
            x_m = x_m[:, :, :, None, :].reshape(B * K * N, 1, -1)
            x_m = self.net_m2pl[layer_id](
                q=x_m,  # [B*K*N, 1, H]
                k=x_pl,  # [B*K*N, M, H]
                v=x_pl,  # [B*K*N, M, H]
                rpe=rpes["rpe_m2pl"],  # [B*K*N, 1, M, H]
                knn_idxs=rpes["knn_idxs_m2pl"],  # [B*K*N, 1, NB_m2pl]
                invalid_mask=invalid_masks[
                    "invalid_mask_m2pl"
                ],  # [B*K*N, 1, M]
            ).reshape(
                B, K, N, -1
            )  # [B, K, N, H]

            # mode to current agent cross attention
            x_m = x_m.reshape(B * K, N, -1)
            x_m = self.net_m2a[layer_id](
                q=x_m,  # [B*K, N, H]
                k=cur_x_a,  # [B*K, N, H]
                v=cur_x_a,  # [B*K, N, H]
                rpe=rpes["rpe_m2a"],  # [B*K, N, N, H]
                knn_idxs=rpes["knn_idxs_m2a"],  # [B*K, N, NB_m2a]
                invalid_mask=invalid_masks["invalid_mask_m2a"],  # [B*K, N, N]
            ).reshape(
                B, K, N, -1
            )  # [B, K, N, H]

        return x_m  # [B, K, N, H]
