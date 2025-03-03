import torch
import torch.nn as nn
import torch.nn.functional as F
from forecast_mae_prediction.src.utils.refactor_na1d import RefactorNeighborhoodAttention1D


class AgentEmbeddingLayer(nn.Module):
    def __init__(
            self,
            in_chans=3,
            embed_dim=32,
            mlp_ratio=3,
            depths=[2, 2, 2],
            num_heads=[2, 4, 8],
            out_indices=[0, 1, 2],
            drop_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.embed = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.num_levels = len(depths)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.out_indices = out_indices

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
            )
            self.levels.append(level)

        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        n = self.num_features[-1]
        self.lateral_convs = nn.ModuleList()
        for i_layer in self.out_indices:
            self.lateral_convs.append(
                nn.Conv1d(self.num_features[i_layer], n, 3, padding=1)
            )
        # for relative positional encoding
        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)
        self.r_t_proj = nn.Linear(4, embed_dim)
        self.r_a2a_proj = nn.Linear(3, embed_dim)


    def forward(self, x):
        """x:[B*N, D, T]"""
        rpes = self.build_rpes(x)
        rpe_a2t = rpes["rpe_a2t"]
        x = x[:, :4, :]
        x = self.embed(x)
        x = x.permute(0, 2, 1) # [B*N, T, D=32]

        out = []
        for idx, level in enumerate(self.levels):
            x, xo, rpe_a2t = level(x, rpe_a2t)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                out.append(x_out.permute(0, 2, 1).contiguous())

        laterals = [
            # idx 0:[B*N, 32, 50] _> [B*N, 128, 50]
            # idx 1:[B*N, 64, 25] -> [B*N, 128, 25]
            # idx 2:[B*N, 128, 13] -> [B*N,128, 13]
            lateral_conv(out[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(out) - 1, 0, -1):
            # F.interpolate() 上采样 这个逻辑是： 将低分辨率信息上采样，最终把所有信息更新到最高分辨率特征中
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                scale_factor=(laterals[i - 1].shape[-1] / laterals[i].shape[-1]),
                mode="linear",
                align_corners=False,
            )
        out = self.fpn_conv(laterals[0])  # [B, D, L]
        return out

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

    def angle_between_2d_vectors(self, ctr_vector: torch.Tensor, nbr_vector: torch.Tensor) -> torch.Tensor:

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


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv1d(
            dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.reduction(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x


class Conv2Downsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.reduction(x)
        x = x.permute(0, 3, 2, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            drop_rate=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = RefactorNeighborhoodAttention1D(
            dim,
            num_heads=num_heads,
            dropout=attn_drop,
        )
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, rpe):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(Q=x, K=x, V=x, rpe=rpe)
        x = self.dropout(x)
        x = shortcut + x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + self.dropout(x)
        return x


class NATBlock(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads,
            downsample=True,
            mlp_ratio=4.0,
            drop=0.0,
            attn_drop=0.0,
            drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_rate=drop_rate[i]
                    if isinstance(drop_rate, list)
                    else drop_rate,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )
        self.rpe_downsample = (
            None if not downsample else Conv2Downsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x, rpe):
        for blk in self.blocks:
            x = blk(x, rpe)
        if self.downsample is None:
            return x, x, rpe
        else:
            B, L, D = x.shape
            x_ = self.downsample(x)
            rpe = rpe.view(B, L, L, D)
            rpe_ = self.rpe_downsample(rpe)
        return x_, x, rpe_
