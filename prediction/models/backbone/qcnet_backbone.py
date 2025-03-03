# Copyright (c) Carizon. All rights reserved.

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from models.base_modules import FourierEmbedding, RPEMHALayer
from utils import angle_between_2d_vectors, wrap_angle


class QCNetBackbone(nn.Module):
    """The QCNet backbone.

    The model is an reimplementation of QCNet: Query-Centric Trajectory
    Prediction.
    """

    def __init__(
        self,
        num_agent_classes: int,
        num_historical_steps: int,
        num_neighbors_pl2pl: int,
        num_neighbors_a2pl: int,
        num_neighbors_a2a: int,
        hidden_dim: int,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        """Initialize method.

        Args:
            num_agent_classes (int): The number of agent classes.
            num_historical_steps (int): The number of historical steps.
            num_neighbors_pl2pl (int): The number of neighbors for polyline to
                polyline attention.
            num_neighbors_a2pl (int): The number of neighbors for agent to
                polyline attention.
            num_neighbors_a2a (int): The number of neighbors for agent to
                agents attention.
            hidden_dim (int): The hidden dimension.
            num_freq_bands (int): The number of frequency bands for fourier
                embedding
            num_map_layers (int): The number of encoder layers for the map
                encoder.
            num_agent_layers (int): The number of encoder layers for the agent
                encoder.
            num_heads (int): The number of heads in the multi-head attention.
            dropout (float): The dropout rate.
        """
        super(QCNetBackbone, self).__init__()
        self.map_encoder = QCNetMapEncoder(
            num_neighbors_pl2pl=num_neighbors_pl2pl,
            num_historical_steps=num_historical_steps,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.agent_encoder = QCNetAgentEncoder(
            num_agent_classes=num_agent_classes,
            num_neighbors_a2pl=num_neighbors_a2pl,
            num_neighbors_a2a=num_neighbors_a2a,
            num_historical_steps=num_historical_steps,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, data: Dict):
        """Forward.

        Args:
            data (dict): The input data.

        Returns:
            scene_encodings (dict): The scene encodings, containing the keys:
                "x_pl": map encodings.
                "x_a": agent encodings.
        """
        scene_enc = {}
        map_enc = self.map_encoder(data)
        scene_enc.update(map_enc)
        agent_enc = self.agent_encoder(data, map_enc)
        scene_enc.update(agent_enc)
        return scene_enc


class QCNetAgentEncoder(nn.Module):
    def __init__(
        self,
        num_agent_classes: int,
        num_historical_steps: int,
        num_neighbors_a2pl: int,
        num_neighbors_a2a: int,
        hidden_dim: int,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        """Initialize method.

        Args:
            num_agent_classes (int): The number of agent classes.
            num_historical_steps (int): The number of historical steps.
            num_neighbors_a2pl (int): The number of neighbors for agents to
                polyline attention.
            num_neighbors_a2a (int): The number of neighbors for agents to
                agents attention.
            hidden_dim (int): The hidden dimension.
            num_freq_bands (int): The number of frequency bands for fourier
                embedding
            num_map_layers (int): The number of encoder layers for the map
                encoder.
            num_agent_layers (int): The number of encoder layers for the agent
                encoder.
            num_heads (int): The number of heads in the multi-head attention.
            dropout (float): The dropout rate.
        """
        super().__init__()

        self.num_agent_classes = num_agent_classes
        self.num_historical_steps = num_historical_steps
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
        # self.emb_dim_x_a = 6
        self.emb_dim_x_a = 4
        self.emb_dim_rpe_t = 4
        self.emb_dim_rpe_a2pl = 3
        self.emb_dim_rpe_a2a = 3

        self.type_a_emb = nn.Embedding(self.num_agent_classes, self.hidden_dim)
        self.x_a_emb = FourierEmbedding(
            input_dim=self.emb_dim_x_a,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=self.emb_dim_rpe_t,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_a2pl_emb = FourierEmbedding(
            input_dim=self.emb_dim_rpe_a2pl,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=self.emb_dim_rpe_a2a,
            hidden_dim=self.hidden_dim,
            num_freq_bands=self.num_freq_bands,
        )

        self.attn_layers_a2t = nn.ModuleList(
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
        self.attn_layers_a2pl = nn.ModuleList(
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
        self.attn_layers_a2a = nn.ModuleList(
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

    def build_queries(self, batch: dict) -> torch.Tensor:
        """
        Build agent quries.

        Args:
            batch (dict): the data dict.

        Returns:
            {
                "x_a": the agent queries. [B, N, T, H]
            }
        """
        his_agent_poses = batch["agent_features"]["gcs"]["poses"]["his"]
        agent_pos = his_agent_poses[..., :2]  # [B, N, T, 2]
        agent_heading = his_agent_poses[..., 2]  # [B, N, T]
        agent_vel = batch["agent_features"]["gcs"]["vels"]["his"][
            ..., :2
        ]  # [B, N, T, 2]
        agent_types = batch["agent_features"]["agent_properties"][
            "classes"
        ]  # [B, N]
        B, N, T, _ = agent_pos.size()
        motion_vec = torch.cat(
            [
                agent_pos.new_zeros(B, N, 1, 2),
                agent_pos[:, :, 1:, :] - agent_pos[:, :, :-1, :],
            ],
            dim=-2,
        )  # [B, N, T, 2], the displacement vector with the first timestep

        # padded with 0
        heading_vec = torch.stack(
            [agent_heading.cos(), agent_heading.sin()], dim=-1
        )  # [B, N, T, 2]

        # x_pt_width = batch["agent_features"]["agent_properties"]["widths"][
        #     :, :, None
        # ].repeat(
        #     1, 1, T
        # )  # [B, N, T]
        # x_pt_length = batch["agent_features"]["agent_properties"]["lengths"][
        #     :, :, None
        # ].repeat(
        #     1, 1, T
        # )  # [B, N, T]
        x_pt_continuous_embs = torch.stack(
            [
                torch.norm(motion_vec, dim=-1),  # [B, N, T]
                angle_between_2d_vectors(heading_vec, motion_vec),  # [B, N, T]
                torch.norm(agent_vel, dim=-1),  # [B, N, T],  # [B, N, T]
                angle_between_2d_vectors(heading_vec, agent_vel),  # [B, N, T]
                # x_pt_width,  # [B, N, T]
                # x_pt_length,  # [B, N, T]
            ],
            dim=-1,
        )  # [B, N, T, 6] -> [B, N, T, 4]
        x_pt_categorical_embs = [
            self.type_a_emb(agent_types.long())[:, :, None, :].repeat(
                1, 1, T, 1
            )
        ]  # [B, N, T, H]
        x_a = self.x_a_emb(
            continuous_inputs=x_pt_continuous_embs,
            categorical_embs=x_pt_categorical_embs,
        ).reshape(
            B, N, T, -1
        )  # [B, N, T, H]
        return {"x_a": x_a}

    def build_rpes(self, batch: dict) -> torch.Tensor:
        """
        Build relative positional encodings.

        Args:
            batch (dict): the data dict.

        Returns:
            {
            "rpe_a2t": rpe for agent to time,  [B*N, T, T, H]
            "rpe_a2pl": rpe for agent to polyline,  [B*T, N, M, H]
            "rpe_a2a": rpe for agent to agent,  [B*T, N, N, H]
            "knn_idxs_a2pl": k-nearest neighbor for agent to polylines
                [B*T, N, NB_a2pl]
            "knn_idxs_a2a": k-nearest neighbor for agent to polylines
                [B*T, N, NB_a2a]
            }
        """
        his_agent_poses = batch["agent_features"]["gcs"]["poses"]["his"]
        agent_pos = his_agent_poses[..., :2]  # [B, N, T, 2]
        agent_heading = his_agent_poses[..., 2]  # [B, N, T]
        pl_poses = batch["map_features"]["gcs"]["pl_poses"]  # [B, M, 3]
        pl_pos = pl_poses[..., :2]  # [B, M, 2]
        pl_heading = pl_poses[..., 2]  # [B, M]
        B, N, T, _ = agent_pos.size()
        M = pl_pos.size(1)

        rel_pos_a2pl = (
            agent_pos[:, :, None, :, :] - pl_pos[:, None, :, None, :]
        )  # [B, N, M, T, 2]
        dist_a2pl = torch.norm(rel_pos_a2pl, dim=-1)  # [B, N, M, T]
        knn_idxs_a2pl = torch.topk(
            -dist_a2pl, self.num_neighbors_a2pl, dim=-2
        ).indices  # [B, N, NB_a2pl, T]
        knn_idxs_a2pl = knn_idxs_a2pl.permute(0, 3, 1, 2)  # [B, T, N, NB_a2pl]
        knn_idxs_a2pl = knn_idxs_a2pl.reshape(
            B * T, N, -1
        )  # [B*T, N, NB_a2pl]

        # build relative positional encoding for agents to polylines
        agent_heading_vec = torch.stack(
            [agent_heading.cos(), agent_heading.sin()], dim=-1
        )  # [B, N, T, 2]
        rel_orient_pl2a = wrap_angle(
            pl_heading[:, None, :, None] - agent_heading[:, :, None, :]
        )  # [B, N, M, T]
        r_a2pl = torch.stack(
            [
                dist_a2pl,  # [B, N, M, T]
                angle_between_2d_vectors(
                    ctr_vector=agent_heading_vec[
                        :, :, None, :, :
                    ],  # [B, N, 1, T, 2]
                    nbr_vector=rel_pos_a2pl,  # [B, N, M, T, 2]
                ),  # [B, N, M, T]
                rel_orient_pl2a,  # [B, N, M, T]
            ],
            dim=-1,
        )  # [B, N, M, T, 3]
        r_a2pl = self.r_a2pl_emb(
            continuous_inputs=r_a2pl, categorical_embs=None
        )  # [B, N, M, T, H]
        r_a2pl = r_a2pl.permute(0, 3, 1, 2, 4).reshape(B * T, N, M, -1)

        # build relative positional encoding for agents to agents
        rel_pos_a2a = (
            agent_pos[:, :, None, :, :] - agent_pos[:, None, :, :, :]
        )  # [B, N, N, T, 2]
        dist_a2a = torch.norm(rel_pos_a2a, dim=-1)  # [B, N, N, T]
        num_a2a_agents = dist_a2a.shape[1]
        knn_idxs_a2a = torch.topk(
            -dist_a2a, min(self.num_neighbors_a2a, num_a2a_agents), dim=-2
        ).indices  # [B, N, NB_a2a, T]
        knn_idxs_a2a = knn_idxs_a2a.permute(0, 3, 1, 2)  # [B, T, N, NB_a2a]
        knn_idxs_a2a = knn_idxs_a2a.reshape(B * T, N, -1)  # [B*T, N, NB_a2a]

        rel_heading_a2a = wrap_angle(
            agent_heading[:, :, None, :] - agent_heading[:, None, :, :]
        )  # [B, N, N, T]
        r_a2a = torch.stack(
            [
                dist_a2a,  # [B, N, N, T]
                angle_between_2d_vectors(
                    ctr_vector=agent_heading_vec[:, :, None, :, :].repeat(
                        1, 1, N, 1, 1
                    ),  # [B, N, N ,T, 2]
                    nbr_vector=rel_pos_a2a,  # [B, N, N, T, 2]
                ),  # [B, N, N, T]
                rel_heading_a2a,  # [B, N, N, T]
            ],
            dim=-1,
        )  # [B, N, N, T, 3]
        r_a2a = self.r_a2a_emb(
            continuous_inputs=r_a2a, categorical_embs=None
        )  # [B, N, N, T, H]
        r_a2a = r_a2a.permute(0, 3, 1, 2, 4).reshape(B * T, N, N, -1)

        # build relative positional encoding for temporal information
        rel_pos_t = (
            agent_pos[:, :, :, None, :] - agent_pos[:, :, None, :, :]
        )  # [B, N, T, T, 2]
        rel_heading_vec = (
            agent_heading_vec[:, :, :, None, :]
            - agent_heading_vec[:, :, None, :, :]
        )  # [B, N, T, T, 2]
        rel_heading_t = wrap_angle(
            agent_heading[:, :, :, None] - agent_heading[:, :, None, :]
        )  # [B, N, T, T]
        t_indices = torch.arange(-T + 1, 1, device=rel_pos_t.device)[
            None, None, :
        ].repeat(B, N, 1)
        rel_indices_t = (
            t_indices[:, :, :, None] - t_indices[:, :, None, :]
        )  # [B, N, T, T]
        r_a2t = torch.stack(
            [
                torch.norm(rel_pos_t, dim=-1),  # [B, N, T, T]
                angle_between_2d_vectors(
                    ctr_vector=rel_heading_vec,
                    nbr_vector=rel_pos_t,
                ),  # [B, N, T, T]
                rel_heading_t,  # [B, N, T, T]
                rel_indices_t,  # [B, N, T, T]
            ],
            dim=-1,
        )  # [B, N, T, T, 4]
        r_a2t = self.r_t_emb(continuous_inputs=r_a2t, categorical_embs=None)
        r_a2t = r_a2t.reshape(B * N, T, T, -1)

        return {
            "rpe_a2t": r_a2t,  # [B*N, T, T, H]
            "rpe_a2pl": r_a2pl,  # [B*T, N, M, H]
            "rpe_a2a": r_a2a,  # [B*T, N, N, H]
            "knn_idxs_a2pl": knn_idxs_a2pl,  # [B*T, N, NB_a2pl]
            "knn_idxs_a2a": knn_idxs_a2a,  # [B*T, N, NB_a2a]
        }

    def build_masks(self, batch: dict) -> Dict[str, torch.Tensor]:
        """
        Build invalid masks for attention.

        Args:
            batch (dict): the data dict.

        Returns:
            {
            "invalid_mask_a2t": the agent to time invalid mask, [B*N, T, T]
            "invalid_mask_a2pl": the agent to polyline invalid mask,
                [B*T, N, M]
            "invalid_mask_a2a": the agent to agent invalid mask, [B*T, N, N]
            }
        """
        agent_state_valid_mask = batch["agent_features"]["state_valid_masks"][
            "his"
        ]  # [B, N, T]
        B, N, T = agent_state_valid_mask.shape
        M = batch["map_features"]["pl_valid_masks"].shape[1]

        a2t_valid_mask = (
            agent_state_valid_mask[..., None]
            .repeat(1, 1, 1, T)
            .reshape(B * N, T, T)
        )  # [B*N, T, T]
        causal_mask = torch.tril(
            torch.ones(T, T, device=a2t_valid_mask.device)
        )  # [T, T]
        a2t_valid_mask = torch.logical_and(
            a2t_valid_mask, causal_mask[None, :, :]
        )  # [B*N, T, T]
        a2t_invalid_mask = ~a2t_valid_mask
        polygon_valid_mask = batch["map_features"]["pl_valid_masks"]  # [B, M]
        a2pl_valid_mask = torch.logical_and(
            agent_state_valid_mask.permute(0, 2, 1)[..., None],  # [B,T,N,1]
            polygon_valid_mask[:, None, None, :],  # [B,1,1,M]
        )
        a2pl_invalid_mask = ~a2pl_valid_mask.reshape(
            B * T, N, M
        )  # [B*T, N, M]
        a2a_valid_mask = agent_state_valid_mask.permute(0, 2, 1)  # [B, T, N]
        a2a_valid_mask = torch.logical_and(
            a2a_valid_mask[:, :, :, None],  # [B, T, N, 1]
            a2a_valid_mask[:, :, None, :],  # [B, T, 1, N]
        )  # [B, T, N, N]
        a2a_invalid_mask = ~a2a_valid_mask.reshape(B * T, N, N)  # [B*T, N, N]

        return {
            "invalid_mask_a2t": a2t_invalid_mask,  # [B*N, T, T]
            "invalid_mask_a2pl": a2pl_invalid_mask,  # [B*T, N, M]
            "invalid_mask_a2a": a2a_invalid_mask,  # [B*T, N, N]
        }

    def forward(
        self,
        batch: dict,
        map_features: dict,
    ) -> Dict[str, torch.torch.Tensor]:
        """Forward.

        Args:
            batch (dict): the input data dict.
            map_features: the input polyline encodings from the map encoder.

        Returns:
            {
                "x_a": the agent queries, [B, N, T, H]
            }
        """
        B, N, T = batch["agent_features"]["state_valid_masks"]["his"].shape
        M = batch["map_features"]["pl_valid_masks"].shape[1]
        queries = self.build_queries(batch)
        rpes = self.build_rpes(batch)
        invalid_masks = self.build_masks(batch)

        x_a = queries["x_a"]
        x_pl = map_features["x_pl"]  # [B, M, H]

        x_pl = (
            x_pl[:, None, :, :].repeat(1, T, 1, 1).reshape(B * T, M, -1)
        )  # [B*T, M, H]
        for i in range(self.num_layers):
            # temporal self-attention
            x_a = x_a.reshape(B * N, T, -1)
            x_a = self.attn_layers_a2t[i](
                q=x_a,  # [B*N, T, H]
                k=x_a,  # [B*N, T, H]
                v=x_a,  # [B*N, T, H]
                rpe=rpes["rpe_a2t"],  # [B*N, T, T, H]
                invalid_mask=invalid_masks["invalid_mask_a2t"],  # [B*N, T, T]
            ).reshape(
                B, N, T, -1
            )  # [B, N, T, H]

            # agents to polylines cross attention
            x_a = x_a.transpose(1, 2).reshape(B * T, N, -1)  # [B*T,N,H]
            x_a = (
                self.attn_layers_a2pl[i](
                    q=x_a,  # [B*T, N, H]
                    k=x_pl,  # [B*T, M, H]
                    v=x_pl,  # [B*T, M, H]
                    rpe=rpes["rpe_a2pl"],  # [B*T, N, M, H]
                    knn_idxs=rpes["knn_idxs_a2pl"],  # [B*T, N, NB_a2pl]
                    invalid_mask=invalid_masks[
                        "invalid_mask_a2pl"
                    ],  # [B*T, N, M]
                )
                .reshape(B, T, N, -1)
                .transpose(1, 2)
            )  # [B, N, T, H]

            # agents to agents self attention
            x_a = x_a.transpose(1, 2).reshape(B * T, N, -1)  # [B*T,N,H]
            x_a = (
                self.attn_layers_a2a[i](
                    q=x_a,  # [B*T, N, H]
                    k=x_a,  # [B*T, N, H]
                    v=x_a,  # [B*T, N, H]
                    rpe=rpes["rpe_a2a"],  # [B*T, N, N, H]
                    knn_idxs=rpes["knn_idxs_a2a"],  # [B*T, N, NB_a2a]
                    invalid_mask=invalid_masks[
                        "invalid_mask_a2a"
                    ],  # [B*T, N, N]
                )
                .reshape(B, T, N, -1)
                .transpose(1, 2)
            )  # [B, N, T, H]

        return {
            "x_a": x_a,  # [B, N, T, H]
        }


class QCNetMapEncoder(nn.Module):
    """Polyline encoder for QCNet."""

    def __init__(
        self,
        num_neighbors_pl2pl: int,
        num_historical_steps: int,
        hidden_dim: int,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        """Initialize method.

        Args:
            hidden_dim (int): The hidden dimension.
            num_freq_bands (int): The number of frequency bands for fourier
                embedding
            num_map_layers (int): The number of encoder layers for the map
                encoder.
            num_agent_layers (int): The number of encoder layers for the agent
                encoder.
            num_heads (int): The number of heads in the multi-head attention.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.num_neighbors_pl2pl = num_neighbors_pl2pl
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_x_pt = 2
        input_dim_x_pl = 1
        input_dim_r_pt2pl = 3
        input_dim_r_pl2pl = 3

        self.type_pl_emb = nn.Embedding(4, hidden_dim)

        self.x_pt_emb = FourierEmbedding(
            input_dim=input_dim_x_pt,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.x_pl_emb = FourierEmbedding(
            input_dim=input_dim_x_pl,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pt2pl_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pl,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pl2pl_emb = FourierEmbedding(
            input_dim=input_dim_r_pl2pl,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.pt2pl_layers = nn.ModuleList(
            [
                RPEMHALayer(
                    d_input=hidden_dim,
                    d_model=head_dim,
                    n_head=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.pl2pl_layers = nn.ModuleList(
            [
                RPEMHALayer(
                    d_input=hidden_dim,
                    d_model=head_dim,
                    n_head=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def build_queries(
        self, batch: dict
    ) -> Tuple[torch.torch.Tensor, torch.torch.Tensor]:
        """Build query embeddings for map polygons and points.

        Note that position and heading information which are defined
        according to coordinate systems should not be included in the query
        embeddings.

        Args:
            batch (dict): the data dict.

        Returns:
            x_pl (tensor): [B, M, hidden_dim]
            x_pt (tensor): [B, M, P, hidden_dim]
        """
        B, M, P = batch["map_features"]["pt_valid_masks"].shape
        x_pl_categorical_embs = [
            self.type_pl_emb(
                batch["map_features"]["pl_types"].long().view(B * M)
            )
        ]
        x_pl = self.x_pl_emb(
            continuous_inputs=None,
            categorical_embs=x_pl_categorical_embs,
        ).view(B, M, -1)

        x_pt_continuous_embs = torch.stack(
            [
                batch["map_features"]["pt_heights"].view(B * M * P),  # added
                batch["map_features"]["pt_magnitudes"].view(B * M * P),
            ],
            dim=-1,
        )  # height mag, [B*M*P, 2]
        x_pt = self.x_pt_emb(
            continuous_inputs=x_pt_continuous_embs,
            categorical_embs=None,
        ).view(B, M, P, -1)
        return x_pl, x_pt

    def build_rpes(
        self, batch: dict
    ) -> Tuple[torch.torch.Tensor, torch.torch.Tensor]:
        """Build relative position embeddings.

        Args:
            batch (dict): the data dict.

        Returns:
            "rpe_pl2pl": rpe for polyline to polyline. Shape [B, M, M, H]
            "rpe_pt2pl": rpe for point to polyline. Shape [B*M, 1, P, H]
            "knn_idxs_pl2pl": knn indices for polyline to polyline.
                Shape [B, M, NB_pl2pl]
        """
        B, M, P = batch["map_features"]["pt_valid_masks"].shape
        pl_heading = batch["map_features"]["gcs"]["pl_poses"][..., 2]  # [B, M]
        orient_pl = torch.stack(
            [
                pl_heading.cos(),
                pl_heading.sin(),
            ],
            dim=-1,
        )  # [B, M, 2]

        # --- construct polygon to polygon relative embeddings ---
        rel_pos_pl2pl = (
            batch["map_features"]["gcs"]["pl_poses"][..., :2][:, None, :]
            - batch["map_features"]["gcs"]["pl_poses"][..., :2][:, :, None]
        )  # [B, M, M, 2]

        # relative heading
        rel_heading_pl2pl = wrap_angle(
            batch["map_features"]["gcs"]["pl_poses"][..., 2][:, None, :]
            - batch["map_features"]["gcs"]["pl_poses"][..., 2][:, :, None]
        )  # [B, M, M]

        dist_pl2pl = torch.norm(rel_pos_pl2pl, dim=-1)  # [B, M, M]
        knn_idxs_pl2pl = torch.topk(
            -dist_pl2pl, self.num_neighbors_pl2pl, dim=-1
        ).indices  # [B, M, NB_pl2pl]

        r_pl2pl = torch.stack(
            [
                dist_pl2pl,  # [B, M, M]
                angle_between_2d_vectors(
                    ctr_vector=orient_pl[:, None, :, :],
                    nbr_vector=rel_pos_pl2pl,
                ),
                rel_heading_pl2pl,
            ],
            dim=-1,
        )  # [B, M, M, 3]
        r_pl2pl = self.r_pl2pl_emb(
            continuous_inputs=r_pl2pl.view(B * M * M, -1),
            categorical_embs=None,
        ).reshape(
            B, M, M, -1
        )  # [B, M, M, H]

        # --- construct point to polygon relative embeddings ---
        # relative distance
        rel_pos_pt2pl = (
            batch["map_features"]["gcs"]["pt_poses"][..., :2]
            - batch["map_features"]["gcs"]["pl_poses"][..., None, :2]
        )  # [B, M, P, 2]

        # relative heading
        rel_heading_pt2pl = wrap_angle(
            batch["map_features"]["gcs"]["pt_poses"][..., 2]
            - batch["map_features"]["gcs"]["pl_poses"][..., None, 2]
        )  # [B, M, P]

        r_pt2pl = torch.stack(
            [
                torch.norm(rel_pos_pt2pl, dim=-1),  # [B, M, P]
                angle_between_2d_vectors(
                    ctr_vector=orient_pl[:, :, None, :],  # [B, M, 1, 2]
                    nbr_vector=rel_pos_pt2pl,  # [B, M, P, 2]
                ),  # [B, M, P]
                rel_heading_pt2pl,  # [B, M, P]
            ],
            dim=-1,
        )  # [B, M, P, 3]
        r_pt2pl = (
            self.r_pt2pl_emb(
                continuous_inputs=r_pt2pl.reshape(B * M * P, -1),
                categorical_embs=None,
            )
            .reshape(B, M, P, -1)
            .reshape(B * M, 1, P, -1)
        )  # [B*M, 1, P, H]

        return {
            "rpe_pl2pl": r_pl2pl,  # [B, M, M, H]
            "rpe_pt2pl": r_pt2pl,  # [B*M, 1, P, H]
            "knn_idxs_pl2pl": knn_idxs_pl2pl,  # [B, M, NB_pl2pl]
        }

    def build_masks(self, batch: dict) -> Dict[str, torch.Tensor]:
        """
        Build invalid masks for attention for x_pt.

        Args:
            batch (dict): the data dict.

        Returns:
            {
            "invalid_mask_a2t": the agent to time invalid mask, [B*N, T, T]
            "invalid_mask_a2pl": the agent to polyline invalid mask,
                [B*T, N, M]
            "invalid_mask_a2a": the agent to agent invalid mask, [B*T, N, N]
            "invalid_mask_pt2pl": the point to polygon invalid mask,
                [B*M, 1, P].
            "invalid_mask_pl2pl": the polygon to polygon invalid mask,
                [B, M, M].
            }
        """
        B, M, P = batch["map_features"]["pt_valid_masks"].shape
        pt_valid_mask = batch["map_features"]["pt_valid_masks"]
        pl_valid_mask = batch["map_features"]["pl_valid_masks"]

        invalid_mask_pt2pl = ~pt_valid_mask.reshape(B * M, 1, P)
        invalid_mask_pl2pl = ~torch.logical_and(
            pl_valid_mask[:, :, None], pl_valid_mask[:, None, :]
        )

        return {
            "invalid_mask_pt2pl": invalid_mask_pt2pl,  # [B*M, 1, P]
            "invalid_mask_pl2pl": invalid_mask_pl2pl,  # [B, M, M]
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.torch.Tensor]:
        """
        Forward function.

        Args:
            batch (dict): the data dict.

        Returns:
            {
            "x_pl": embedding of the polygon features [B, M, P, D].
            }
        """
        x_pl, x_pt = self.build_queries(batch)
        rpes = self.build_rpes(batch)
        invalid_masks = self.build_masks(batch)
        B, M, P, _ = x_pt.shape
        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i](
                q=x_pl.reshape(B * M, 1, -1),  # [B*M, 1, H]
                k=x_pt.reshape(B * M, P, -1),  # [B*M, P, H]
                v=x_pt.reshape(B * M, P, -1),  # [B*M, P, H]
                rpe=rpes["rpe_pt2pl"],  # [B*M, 1, P, H]
                invalid_mask=invalid_masks[
                    "invalid_mask_pt2pl"
                ],  # [B*M, 1, P]
            ).reshape(
                B, M, -1
            )  # [B, M, H]
            x_pl = self.pl2pl_layers[i](
                q=x_pl,  # [B, M, H]
                k=x_pl,  # [B, M, H]
                v=x_pl,  # [B, M, H]
                rpe=rpes["rpe_pl2pl"],  # [B, M, M, H]
                knn_idxs=rpes["knn_idxs_pl2pl"],  # [B, M, NB_pl2pl]
                invalid_mask=invalid_masks["invalid_mask_pl2pl"],  # [B, M, M]
            )  # [B, M, H]

        return {"x_pl": x_pl}
