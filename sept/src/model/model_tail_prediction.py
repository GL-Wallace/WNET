from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.obs_embedding import AgentEmbeddingLayer
from .layers.pline_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block


# pre-train: Tail Prediction from paper SEPT
class ModelTailPrediction(nn.Module):
    def __init__(
            self,
            embed_dim=128,
            encoder_depth=3,
            decoder_depth=3,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_path=0.2,
            actor_mask_ratio: float = 0.5,
            lane_mask_ratio: float = 0.5,
            history_steps: int = 10,
            future_steps: int = 30,
            loss_weight: List[float] = [1.0, 1.0, 0.35],
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.actor_mask_ratio = actor_mask_ratio
        self.lane_mask_ratio = lane_mask_ratio
        self.loss_weight = loss_weight
        self.traj_embed = AgentEmbeddingLayer(4, 32, drop_rate=drop_path)
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

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.actor_type_embed = nn.Parameter(torch.Tensor(16, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, 1, embed_dim))
        self.head_embed = nn.Linear(2560, embed_dim)
        self.tail_pred = nn.Linear(embed_dim, 30 * 2)
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

    @staticmethod
    def agent_random_masking(
            traj_tokens, traj_padding_mask, mask_ratio
    ):
        # 构建head-tail两部分数据, 按照mask ratio, 返回head tokens and head padding_mask 用于 tail 预测
        B, N, T, D = traj_tokens.shape  # [B, N, T, D_128]
        TS_keeps = int(T * (1 - mask_ratio))
        head_tokens = traj_tokens[:, :, :TS_keeps, :]
        head_key_padding_mask = traj_padding_mask.any(dim=2)
        # head_key_padding_mask = torch.zeros(~traj_padding_mask[:, : ], device=traj_tokens.device)

        return (
            head_tokens,
            head_key_padding_mask,
        )

    def forward(self, data):
        hist = data['x']
        fut = data['y']
        # traj = torch.cat((hist, fut), dim=2)
        traj = hist[:, :, :20, :]
        traj_padding_mask = data["x_padding_mask"]
        traj_feat = torch.cat(  # [B, N, T, 5]
            [
                traj,
                data["x_velocity"][:, :, :20][..., None],
                ~traj_padding_mask[:, :, :20][..., None],
                data["x_angles"][:, :, :20][..., None]
            ],
            dim=-1,
        )

        B, N, T, D = traj_feat.shape  # B: Batch size, N: number of obs, T: Time Step, D: Dimension
        traj_feat = traj_feat.view(B * N, T, D)  # [B*N, L, 5]
        traj_feat = traj_feat.permute(0, 2, 1).contiguous()  # [B*N, 5, L]
        traj_feat = self.traj_embed(traj_feat)
        # traj_feat = traj_feat.view(B, N, traj_feat.shape[-1])  # 1, 16, 128
        traj_feat = traj_feat.view(B, N, T, -1)  # [B, N, L, 128]

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)  # [1, 53, 20, 3]
        # B: Batch size, M: number of polylines, T: number of waypoints, D: Dimension
        B, M, L, D = lane_feat.shape
        lane_feat = lane_feat.view(-1, L, D).contiguous()
        lane_feat = self.lane_embed(lane_feat)  # [53, 128] b,c
        lane_feat = lane_feat.view(B, M, L, -1)  # [B, M, L, 128]

        actor_type_embed = data["x_attr"][..., 0].long()
        actor_type_embed = self.actor_type_embed[actor_type_embed]
        actor_type_embed = actor_type_embed[:, :, None, :].repeat(1, 1, T, 1)
        traj_feat += actor_type_embed
        lane_feat += self.lane_type_embed

        x_centers = torch.cat(  # [B, N+M, 2]
            [data["x_centers"], data["lane_centers"]], dim=1
        )
        angles = torch.cat(  # [B, N+M]
            [
                data["x_angles"][..., 19],
                data["lane_angles"],
            ],
            dim=1,
        )
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [B, N+M, 2]
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)

        pos_embed = self.pos_embed(pos_feat)  # [B, N+M, D]
        traj_feat += pos_embed[:, :N, None, :].repeat(1, 1, T, 1)  # [B, N, T, D_128]
        lane_feat += pos_embed[:, -M:, None, :].repeat(1, 1, L, 1)  # [B, M, L, D_128]

        (
            head_tokens,
            head_key_padding_mask,
        ) = self.agent_random_masking(
            traj_feat,
            traj_padding_mask,
            self.actor_mask_ratio,
        )
        head_tokens = head_tokens.mean(dim=2)  # [B, N, D]将所有的time step的特征进行平均, 从而压缩成一个聚合信息.
        # head_tokens = head_tokens.view(B, N, -1)
        # head_tokens = self.head_embed(head_tokens)
        lane_tokens = lane_feat.mean(dim=2)
        lane_key_padding_mask = lane_padding_mask.any(dim=2)

        x = torch.cat(
            [head_tokens, lane_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [head_key_padding_mask, lane_key_padding_mask], dim=1,
        )

        # encoding
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # decoding
        x_decoder = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        tail_token = x_decoder[:, :N].reshape(-1, self.embed_dim)

        # Tail pred loss from [B*N, 128]-> [B*N, 40]-> [B*N, 20, 2]
        y_hat = self.tail_pred(tail_token).view(-1, 30, 2)
        y = hist[:, :, 20:, :].view(-1, 30, 2)
        reg_mask = ~data["x_padding_mask"][:, :, 20:50]
        reg_mask = reg_mask.view(-1, 30)
        output_result = y_hat[reg_mask]
        tail_loss = F.mse_loss(y_hat[reg_mask], y[reg_mask])
        out = {
            "loss": tail_loss,
            "output_result": output_result,
        }
        return out


# pre-train: Tail Prediction from paper SEPT
class ModelTailPredictionV1(nn.Module):
    def __init__(
            self,
            embed_dim=128,
            encoder_depth=3,
            decoder_depth=3,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_path=0.2,
            actor_mask_ratio: float = 0.5,
            lane_mask_ratio: float = 0.5,
            history_steps: int = 10,
            future_steps: int = 30,

            loss_weight: List[float] = [1.0, 1.0, 0.35],
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.actor_mask_ratio = actor_mask_ratio
        self.lane_mask_ratio = lane_mask_ratio
        self.loss_weight = loss_weight

        self.traj_embed = AgentEmbeddingLayer(4, 32, drop_rate=drop_path)
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

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(decoder_depth)
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.actor_type_embed = nn.Parameter(torch.Tensor(16, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, 1, embed_dim))
        self.head_embed = nn.Linear(2560, embed_dim)
        self.tail_pred = nn.Linear(embed_dim, 20 * 2)
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

    @staticmethod
    def agent_random_masking(
            traj_tokens, traj_padding_mask, mask_ratio
    ):
        # 构建head-tail两部分数据, 按照mask ratio, 返回head tokens and head padding_mask 用于 tail 预测
        B, N, T, D = traj_tokens.shape  # [B, N, T, D_128]
        TS_keeps = int(T * (1 - mask_ratio))
        head_tokens = traj_tokens[:, :, :TS_keeps, :]
        head_key_padding_mask = traj_padding_mask.any(dim=2)
        # head_key_padding_mask = torch.zeros(~traj_padding_mask[:, : ], device=traj_tokens.device)

        return (
            head_tokens,
            head_key_padding_mask,
        )

    def forward(self, data):
        traj = torch.cat((data['x'], data['y']), dim=2)
        traj_padding_mask = data["x_padding_mask"]
        traj_feat = torch.cat(  # [B, N, T, 5]
            [
                traj,
                data["x_velocity"][..., None],
                ~traj_padding_mask[..., None],
                data["x_angles"][..., None]
            ],
            dim=-1,
        )

        (
            head_tokens,
            head_key_padding_mask,
        ) = self.agent_random_masking(
            traj_feat,
            traj_padding_mask,
            self.actor_mask_ratio,
        )
        B, N, T, D = head_tokens.shape  # B: Batch size, N: number of obs, T: Time Step, D: Dimension [B, N, T, D]
        head_tokens = head_tokens.permute(0, 3, 1, 2).contiguous() # [B, D, N, T]
        traj_feat = self.traj_embed(head_tokens)
        # traj_feat = traj_feat.view(B, N, traj_feat.shape[-1])  # 1, 16, 128
        traj_feat = traj_feat.view(B, N, T, -1)  # [B, N, L, 128]

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1)  # [1, 53, 20, 3]
        # B: Batch size, M: number of polylines, T: number of waypoints, D: Dimension
        B, M, L, D = lane_feat.shape
        lane_feat = lane_feat.view(-1, L, D).contiguous()
        lane_feat = self.lane_embed(lane_feat)  # [53, 128] b,c
        lane_feat = lane_feat.view(B, M, L, -1)  # [B, M, L, 128]

        actor_type_embed = data["x_attr"][..., 0].long()
        actor_type_embed = self.actor_type_embed[actor_type_embed]
        actor_type_embed = actor_type_embed[:, :, None, :].repeat(1, 1, T, 1)
        traj_feat += actor_type_embed
        lane_feat += self.lane_type_embed

        x_centers = torch.cat(  # [B, N+M, 2]
            [data["x_centers"], data["lane_centers"]], dim=1
        )
        angles = torch.cat(  # [B, N+M]
            [
                data["x_angles"][..., 9],
                data["lane_angles"],
            ],
            dim=1,
        )
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [B, N+M, 2]
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)

        pos_embed = self.pos_embed(pos_feat)  # [B, N+M, D]
        traj_feat += pos_embed[:, :N, None, :].repeat(1, 1, T, 1)  # [B, N, T, D_128]
        lane_feat += pos_embed[:, -M:, None, :].repeat(1, 1, L, 1)  # [B, M, L, D_128]

        # (
        #     head_tokens,
        #     head_key_padding_mask,
        # ) = self.agent_random_masking(
        #     traj_feat,
        #     traj_padding_mask,
        #     self.actor_mask_ratio,
        # )
        head_tokens = head_tokens.mean(dim=2)  # [B, N, D]将所有的time step的特征进行平均, 从而压缩成一个聚合信息.
        # head_tokens = head_tokens.view(B, N, -1)
        # head_tokens = self.head_embed(head_tokens)
        lane_tokens = lane_feat.mean(dim=2)
        lane_key_padding_mask = lane_padding_mask.any(dim=2)

        x = torch.cat(
            [head_tokens, lane_tokens], dim=1
        )
        key_padding_mask = torch.cat(
            [head_key_padding_mask, lane_key_padding_mask], dim=1,
        )

        # encoding
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # decoding
        x_decoder = self.decoder_embed(x)
        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        tail_token = x_decoder[:, :N].reshape(-1, self.embed_dim)

        # Tail pred loss # from  [B*N, 128]-> [B*N, 40]-> [B*N, 20, 2]
        y_hat = self.tail_pred(tail_token).view(-1, 20, 2)
        y = traj[:, :, 20:, :].view(-1, 20, 2)
        reg_mask = ~data["x_padding_mask"][:, :, 20:]
        reg_mask = reg_mask.view(-1, 20)
        output_result = y_hat[reg_mask]
        tail_loss = F.mse_loss(y_hat[reg_mask], y[reg_mask])
        out = {
            "loss": tail_loss,
            "output_result": output_result,
        }

        # if not self.training:
        #     out["x_hat"] = x_hat.view(B, N, 10, 2)
        #     out["y_hat"] = y_hat.view(1, B, N, 30, 2)
        #     out["lane_hat"] = lane_pred.view(B, M, L, 2)
        #     out["lane_keep_ids"] = lane_ids_keep_list
        #     out["hist_keep_ids"] = hist_keep_ids_list
        #     out["fut_keep_ids"] = fut_keep_ids_list

        return out
