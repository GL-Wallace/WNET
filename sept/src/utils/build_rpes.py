import math
from symtable import Class
from typing import List, Optional
import torch
import torch.nn as nn


class BuildRPES:
    def __init__(self, embed_dim=32):
        self.r_a2a_proj = nn.Linear(3, embed_dim)
        pass

    def build_rpes(self, x) -> torch.Tensor:
        """
        Build relative positional encodings.

        Args:
            x (tensor): the data dict.

        Returns:
            {
            "rpe_a2t": rel_pe for agent to time,  [B*N, T, T, H]
            "rpe_a2a": rpe for agent to agent,  [B*T, N, N, H]
            "knn_idxs_a2a": k-nearest neighbor for agent to agent [T, N, NB_a2a]
            }
        """

        his_agent_poses = x.permute(0, 2, 1)  # [b * n, T, 2]
        agent_pos = his_agent_poses[..., :2]  # [B * N, T, 2]
        agent_heading = his_agent_poses[..., 4]  # [B * N, T]
        BN, T, _ = agent_pos.size()

        # build relative positional encoding for agents to agents
        rel_pos_a2a = (
                agent_pos[:, None, :, :] - agent_pos[None, :, :, :]
        )  # [b * n, b * n, T, 2]

        dist_a2a = torch.norm(rel_pos_a2a, dim=-1)  # [b * n, b * n, T]

        num_a2a_agents = dist_a2a.shape[1]
        num_neighbors_a2a = int(num_a2a_agents / 3)
        knn_idxs_a2a = torch.topk(
            -dist_a2a, num_neighbors_a2a, dim=-2
        ).indices  # [b * n, NB_a2a, T]
        knn_idxs_a2a = knn_idxs_a2a.permute(2, 0, 1)  # [T, b * n, NB_a2a]
        knn_idxs_a2a = knn_idxs_a2a.reshape(T, BN, -1)  # [T, b * n, NB_a2a]

        rel_heading_a2a = agent_heading[:, None, :] - agent_heading[None, :, :]  # [b * n, b * n, T]
        rel_heading_a2a = (rel_heading_a2a + torch.pi) % (2 * torch.pi) - torch.pi
        r_a2a = torch.stack(
            [
                dist_a2a,  # [b * n, b * n, T]
                self.angle_between_2d_vectors(
                    ctr_vector=rel_heading_a2a[:, :, :, None].repeat(
                        1, 1, 1, 2
                    ),  # [b * n, b * n, T, 2]
                    nbr_vector=rel_pos_a2a,  # [b * n, b * n, T, 2]
                ),  # [b * n, b * n, T]
                rel_heading_a2a,  # [b * n, b * n, T]
            ],
            dim=-1,
        )  # [b * n, b * n, T, 3]
        # 假设 r_a2a 的输入维度为 [b * n, b * n, T, 原始维度]
        # [b * n, b * n, T, H]
        # r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        r_a2a = self.r_a2a_proj(r_a2a)
        r_a2a = r_a2a.permute(0, 2, 1, 3).reshape(BN * T, BN, -1)

        # build relative positional encoding for temporal information
        rel_pos_t = (
                agent_pos[:, :, None, :] - agent_pos[:, None, :, :]
        )  # [b * n, T, T, 2]
        rel_heading_vec = (
            torch.stack((torch.cos(agent_heading), torch.sin(agent_heading)), dim=-1)
        )  # [b * n, T, 2]
        rel_heading_vec = (
                rel_heading_vec[:, :, None, :] - rel_heading_vec[:, None, :, :]
        )  # [b * n, T, T, 2]
        rel_heading_t = agent_heading[:, :, None] - agent_heading[:, None, :]
        rel_heading_t = torch.atan2(torch.sin(rel_heading_t), torch.cos(rel_heading_t))  # [b * n, T, T]

        t_indices = torch.arange(-T + 1, 1, device=rel_pos_t.device).repeat(BN, 1)  # [b * n, T]
        rel_indices_t = (
                t_indices[:, :, None] - t_indices[:, None, :]
        )  # [b * n, T, T]

        r_a2t = torch.stack(
            [
                torch.norm(rel_pos_t, dim=-1),  # [b * n, T, T]
                self.angle_between_2d_vectors(
                    ctr_vector=rel_heading_vec,
                    nbr_vector=rel_pos_t,
                ),  # [b * n, T, T]
                rel_heading_t,  # [b * n, T, T]
                rel_indices_t,  # [b * n, T, T]
            ],
            dim=-1,
        )  # [b * n, T, T, 4]

        # r_a2t = self.r_t_emb(continuous_inputs=r_a2t, categorical_embs=None)
        r_a2t = self.r_t_proj(r_a2t)
        r_a2t = r_a2t.reshape(BN * T, T, -1)

        return {
            "rpe_a2t": r_a2t,  # [B*N, T, T, H]
            "rpe_a2a": r_a2a,  # [B*T, N, N, H]
            "knn_idxs_a2a": knn_idxs_a2a,  # [T, N, NB_a2a]
        }

    def build_a2a_rpes(self, x) -> torch.Tensor:
        # build relative positional encoding for agents to agents
        his_agent_poses = x.permute(0, 2, 3, 1)  # [B, N, T, 2]
        agent_pos = his_agent_poses[..., :2]  # [B , N, T, 2]
        agent_heading = his_agent_poses[..., 4]  # [B , N, T]

        # 相对夹角
        agent_heading_vec = torch.stack(
            [agent_heading.cos(), agent_heading.sin()], dim=-1
        )  # [B, N, T, 2]
        B, N, T, _ = agent_pos.size()

        # 计算a2a相对位置,
        rel_pos_a2a = (
                agent_pos[:, :, None, :, :] - agent_pos[:, None, :, :, :]
        )  # [B, N, N, T, 2]

        # 计算a2a相对距离
        dist_a2a = torch.norm(rel_pos_a2a, dim=-1)  # [B, N, N, T]

        # 计算a2a相对角度, 真实角度
        rel_heading_a2a = agent_heading[:, :, None, :] - agent_heading[:, None, :, :]
        # [B, N, N, T]
        rel_heading_a2a = (rel_heading_a2a + torch.pi) % (2 * torch.pi) - torch.pi

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


def angle_between_2d_vectors(ctr_vector: torch.Tensor, nbr_vector: torch.Tensor) -> torch.Tensor:
    """Calculate the angle between two 2D vectors in radiant.

    Args:
        ctr_vector(torch.Tensor): The 2D vector chosen to be centered.
        nbr_vector(torch.Tensor): The 2D vector chosen to be the neighbor.
    Returns:
        torch.Tensor: The angle between the vectors in radians.
    """
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1]
        - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )

class FourierEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_freq_bands: int,
    ) -> None:
        """Initialize.

        Args:
            input_dim (int): the dimension of the input data.
            hidden_dim (int): the dimension of the hidden layer.
            num_freq_bands (int): the number of frequency bands.
        """
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = (
            nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        )
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            continuous_inputs (Optional[torch.Tensor]): the continuous features
                to be embedded. Shape [..., feat_dim].
            categorical_embs (Optional[List[torch.Tensor]]): the categorical
                features to be embedded. The input categorical embeddings
                be output of nn.Embedding layers. Shape [[..., emb_dim], ...,
                [..., emb_dim]].
        """
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError(
                    "Both continuous_inputs and categorical_embs are None"
                )
        else:
            x = (
                continuous_inputs.unsqueeze(-1)
                * self.freqs.weight
                * 2
                * math.pi
            )
            x = torch.cat(
                [x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1
            )
            continuous_embs: List[Optional[torch.Tensor]] = [
                None
            ] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[..., i, :])
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return self.to_out(x)



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