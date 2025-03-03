from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder
from .layers.transformer_blocks import Block


class ModelForecast(nn.Module):
    def __init__(
            self,
            embed_dim: object = 128,
            encoder_depth: object = 4,
            num_heads: object = 8,
            mlp_ratio: object = 4.0,
            qkv_bias: object = False,
            drop_path: object = 0.2,
            future_steps: int = 30,
            k: int = 6,
    ) -> None:
        super().__init__()
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(16, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.decoder = MultimodalDecoder(embed_dim, future_steps, k)
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, future_steps * 2)
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net."):]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :10]
        hist_key_padding_mask = data["x_key_padding_mask"]  # [1, 16]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)  # [16, 50, 4]
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)  # [1*16]
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])
        # 创建actor_feat_temp的目的是什么？ 为什么需要通过padding的位置进行这种维度的变化。
        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)  # [1,69,2]
        angles = torch.cat([data["x_angles"][:, :, 9], data["lane_angles"]], dim=1)  # [1, 69]
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [1,69,2]
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)  # [1, 69, 4]
        pos_embed = self.pos_embed(pos_feat)  # [1, 69, 128]

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 0].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        x_agent = x_encoder[:, 0]
        y_hat, pi = self.decoder(x_agent)  # 选择每个批次中的第一个序列元素 【1, 69, 128】 表示，【1, 128】69的第一个

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, 30, 2)

        return {
            "y_hat": y_hat,
            "pi": pi,
            "y_hat_others": y_hat_others,
        }
