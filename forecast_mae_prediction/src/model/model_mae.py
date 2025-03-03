from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block


class ModelMAE(nn.Module):
    def __init__(
            self,
            embed_dim=128,
            encoder_depth=4,
            decoder_depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_path=0.2,
            actor_mask_ratio: float = 0.5,
            lane_mask_ratio: float = 0.5,
            history_steps: int = 10,
            future_steps: int = 30,
            max_lane_pred_dim: int = 35,
            loss_weight: List[float] = [1.0, 1.0, 0.35],
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.actor_mask_ratio = actor_mask_ratio
        self.lane_mask_ratio = lane_mask_ratio
        self.loss_weight = loss_weight
        self.max_lane_pred_dim = max_lane_pred_dim
        self.hist_embed = AgentEmbeddingLayer(4, 32, drop_path_rate=drop_path)
        self.future_embed = AgentEmbeddingLayer(3, 32, drop_path_rate=drop_path)
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
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.lane_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.future_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.history_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.future_pred = nn.Linear(embed_dim, future_steps * 2)
        self.history_pred = nn.Linear(embed_dim, history_steps * 2)
        self.lane_pred = nn.Linear(embed_dim, max_lane_pred_dim * 2)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)
        nn.init.normal_(self.future_mask_token, std=0.02)
        nn.init.normal_(self.lane_mask_token, std=0.02)
        nn.init.normal_(self.history_mask_token, std=0.02)

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
            hist_tokens, fut_tokens, mask_ratio, future_padding_mask, num_actors
    ):
        # [1, 16, 60] dim=-1, all（全真为True， 否则为false）取反，全真为false， 全假为真（这是都有数据）
        pred_masks = ~future_padding_mask.all(-1)  # [B, A] .all(-1)_>[1, 16]{全无信息true， 部分有信息false，全有信息false} 取反-》{全无信息false， 只要有信息就是true}
        fut_num_tokens = pred_masks.sum(-1)  # [B] 16

        len_keeps = (fut_num_tokens * (1 - mask_ratio)).int() # 0.5*16.int() = 8
        hist_masked_tokens, fut_masked_tokens = [], []
        hist_keep_ids_list, fut_keep_ids_list = [], []
        hist_key_padding_mask, fut_key_padding_mask = [], []

        device = hist_tokens.device
        # hist_tokens.shape[1] = 16;                      arange(16) 生成一个length 16的tensor
        agent_ids = torch.arange(hist_tokens.shape[1], device=device) #
        for i, (fut_num_token, len_keep, future_pred_mask) in enumerate(
                zip(fut_num_tokens, len_keeps, pred_masks)
        ):
            pred_agent_ids = agent_ids[future_pred_mask] # [ future_pred_mask shape[16] ]通过有未来轨迹的actor 来保存车的ID
            noise = torch.rand(fut_num_token, device=device) # fut_num_token: 16 ;
            ids_shuffle = torch.argsort(noise) # 返回一个与 noise 张量形状相同的张量 ids_shuffle， 其中的每个元素是对应的【noise tensor元素升序排序后』的idx
            fut_ids_keep = ids_shuffle[:len_keep] # 选取后8个id 作为留下的id
            fut_ids_keep = pred_agent_ids[fut_ids_keep]
            fut_keep_ids_list.append(fut_ids_keep)
            hist_keep_mask = torch.zeros_like(agent_ids) # [16]
            hist_keep_mask = hist_keep_mask.bool()
            hist_keep_mask[: num_actors[i]] = True # 所有的位置都为true, num_actors = 16
            hist_keep_mask[fut_ids_keep] = False # 对刚才选出来的8个id设为false
            hist_ids_keep = agent_ids[hist_keep_mask] # 保存的是不包含future保存的ids 【8】
            hist_keep_ids_list.append(hist_ids_keep)

            fut_masked_tokens.append(fut_tokens[i, fut_ids_keep])  # [1, 16, 128]  [8]
            hist_masked_tokens.append(hist_tokens[i, hist_ids_keep])
# future 和 history 保存的id正好相反，随机取的0.5*actors，保存在了future中，相反的保存在了history中
            fut_key_padding_mask.append(torch.zeros(len_keep, device=device))
            hist_key_padding_mask.append(torch.zeros(len(hist_ids_keep), device=device))
        # pad_sequence 函数位于 torch.nn.utils.rnn 模块中，将 fut_masked_tokens 列表中的张量填充到相同长度，并将批次维度放在第一维。
        fut_masked_tokens = pad_sequence(fut_masked_tokens, batch_first=True)
        hist_masked_tokens = pad_sequence(hist_masked_tokens, batch_first=True)
        fut_key_padding_mask = pad_sequence(
            fut_key_padding_mask, batch_first=True, padding_value=True
        )
        hist_key_padding_mask = pad_sequence(
            hist_key_padding_mask, batch_first=True, padding_value=True
        )

        return (
            hist_masked_tokens,
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        )

    @staticmethod
    def lane_random_masking(x, future_mask_ratio, key_padding_mask):
        num_tokens = (~key_padding_mask).sum(1)  # (B, ) # 实际有效的waypoints数量
        len_keeps = torch.ceil(num_tokens * (1 - future_mask_ratio)).int() # torch.ceil向上取整27

        x_masked, new_key_padding_mask, ids_keep_list = [], [], []
        for i, (num_token, len_keep) in enumerate(zip(num_tokens, len_keeps)):
            noise = torch.rand(num_token, device=x.device)
            ids_shuffle = torch.argsort(noise)
            # 随机了选择27个id，作为保存下来的lane ids
            ids_keep = ids_shuffle[:len_keep]
            ids_keep_list.append(ids_keep)
            x_masked.append(x[i, ids_keep]) # masked就是0-27
            new_key_padding_mask.append(torch.zeros(len_keep, device=x.device)) # 后面拼接上27全0矩阵

        x_masked = pad_sequence(x_masked, batch_first=True)
        new_key_padding_mask = pad_sequence(
            new_key_padding_mask, batch_first=True, padding_value=True
        )

        return x_masked, new_key_padding_mask, ids_keep_list

    def forward(self, data):
        # x_padding_mask: [1, 16, 110], batch size: 1，
        hist_padding_mask = data["x_padding_mask"][:, :, :10]
        hist_feat = torch.cat(  # [1, N, TS , 4 ]
            [
                data["x"],  # [1, 16, 10, 2]
                # ..., 选择数组的所有现有轴, None在所选位置添加一个新的轴】
                data["x_velocity_diff"][..., None],  # [1, 16, 50，1]
                ~hist_padding_mask[..., None],  # [1, 16, 50, 1] padding 是有数据的位置为false，取反则为true []
            ],
            dim=-1,
        )
        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)  # [16, 50, 4]
        hist_feat = hist_feat.permute(0, 2, 1).contiguous() # [N, 4, 50]
        hist_feat = self.hist_embed(hist_feat) # [16, 128]
        hist_feat = hist_feat.view(B, N, hist_feat.shape[-1]) # 1, 16, 128

        # data["x_padding_mask"] [1, 16, 110]
        future_padding_mask = data["x_padding_mask"][:, :, 10:] # [1, 16, 60]
        # [1, 16, 60, 2] + ~[1, 16, 60, 1] -> [1, 16, 60, 3]
        future_feat = torch.cat([data["y"], ~future_padding_mask[..., None]], dim=-1)
        B, N, L, D = future_feat.shape
        future_feat = future_feat.view(B * N, L, D) # [16, 60, 3]
        # input[16, 3, 60] -> output: [1, 16, 128]
        future_feat = self.future_embed(future_feat.permute(0, 2, 1).contiguous())
        future_feat = future_feat.view(B, N, future_feat.shape[-1])

        # []
        lane_padding_mask = data["lane_padding_mask"]
        # [1, 53, 20, 2] - [1, 53, 1, 2] 每个车道位置相对于车道中心点的偏移量 [b, n, waypoints, d]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_feat = torch.cat([lane_normalized, ~lane_padding_mask[..., None]], dim=-1) # [1, 53, 20, 3]
        B, M, L, D = lane_feat.shape # b:1, m:53, L: 20, d:3
        lane_feat =  lane_feat.view(-1, L, D).contiguous() # (53, 20, 3)
        lane_feat = self.lane_embed(lane_feat) # [53, 128] b,c
        lane_feat = lane_feat.view(B, M, -1) # 1, 53, 128
        # x_attr[1, 16, 3][..., 2], ...:表示选择所有前置维度, 2: 表示在最后一个维度上选择索引为 2 的元素; dim:2: OBJECT_TYPE_MAP_COMBINED
        actor_type_embed = data["x_attr"][..., 0].long() # [1, 16]
        # actor_type_embed 是一个形状为 [1, 16] 的张量，包含了 16 个索引值。
        # 假设: actor_type_embed = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
        # self.actor_type_embed 是一个形状为 [4, 128] 的张量，其中 128 是嵌入向量的维度。     # self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        #当我们使用 actor_type_embed 对 self.actor_type_embed 进行索引时，
        # PyTorch 会将 actor_type_embed 中的每个索引值替换为 self.actor_type_embed 中对应的嵌入向量。
        # actor_type_embed[0, 0] 是 0，所以 self.actor_type_embed[0] 会被提取。
        # actor_type_embed[0, 1] 是 1，所以 self.actor_type_embed[1] 会被提取。依此类推。最终，actor_type_embed 中的每个索引值都将被替换为相应的嵌入向量。
        actor_type_embed = self.actor_type_embed[actor_type_embed] #[1, 16, 128]
        # 用于表示 4 种不同类型的 OBJECT_TYPE_MAP_COMBINED 的嵌入向量。每种类型的 OBJECT_TYPE_MAP_COMBINED 都有一个对应的嵌入向量，这些嵌入向量在训练过程中会被优化，以便更好地表示不同类型的 OBJECT_TYPE_MAP_COMBINED。
        # self.actor_type_embed 其中的值是随机参数, self.lane_type_embed 都是随机参数,作为输入加入到模型中的意义是什么?
        # 1. 避免零梯度 2. 合适的随机初始化（例如 Xavier 初始化或 He 初始化）可以使得网络在训练初期更快地收敛,
        # 3.随机初始化可以增加模型的多样性，使得模型在训练过程中能够探索更多的解空间
        # 1,16,128】【1, 53, 128, + 1,1,128】 【1, 16,128】
        hist_feat += actor_type_embed
        lane_feat += self.lane_type_embed
        future_feat += actor_type_embed

        x_centers = torch.cat( # 16+16+53, [1, 85, 2]
            [data["x_centers"], data["x_centers"], data["lane_centers"]], dim=1
        )
        angles = torch.cat( # [1, 85]
            [
                data["x_angles"][..., 9],
                data["x_angles"][..., 9],
                data["lane_angles"],
            ],
            dim=1,
        )
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) # why cos, sin?? [1, 85, 2]
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)  # [1, 85, 4]

        pos_embed = self.pos_embed(pos_feat) # [1, 85, 128]
        hist_feat += pos_embed[:, :N] # [1, 16, 128] 把前16个的actor的位置和heading给了hist——feature
        lane_feat += pos_embed[:, -M:] # [1, 53, 128]
        future_feat += pos_embed[:, N: N + N] # [1, 16, 128]

        (
            hist_masked_tokens, # [1, 8, 128]
            hist_keep_ids_list,
            hist_key_padding_mask,
            fut_masked_tokens,
            fut_keep_ids_list,
            fut_key_padding_mask,
        ) = self.agent_random_masking(
            hist_feat,
            future_feat,
            self.actor_mask_ratio,
            future_padding_mask,
            data["num_actors"],
        )
        #
        lane_mask_ratio = self.lane_mask_ratio
        (
            lane_masked_tokens,
            lane_key_padding_mask,
            lane_ids_keep_list,
        ) = self.lane_random_masking(
            lane_feat, lane_mask_ratio, data["lane_key_padding_mask"]
        )

        x = torch.cat( # [1, 43, 128]
            [hist_masked_tokens, fut_masked_tokens, lane_masked_tokens], dim=1
        )
        key_padding_mask = torch.cat( # 这三个都是全0矩阵，[1, 43]
            [hist_key_padding_mask, fut_key_padding_mask, lane_key_padding_mask],
            dim=1,
        )

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # decoding
        x_decoder = self.decoder_embed(x)
        Nh, Nf, Nl = (
            hist_masked_tokens.shape[1], # 8
            fut_masked_tokens.shape[1],  # 8
            lane_masked_tokens.shape[1], # 27
        ) # assert condition, "Optional error message"
        assert x_decoder.shape[1] == Nh + Nf + Nl # 是一种用于调试和验证代码正确性的工具。它的作用是检查一个条件是否为真，如果该条件为假，程序会抛出一个 AssertionError 并终止执行。
        hist_tokens = x_decoder[:, :Nh]
        fut_tokens = x_decoder[:, Nh: Nh + Nf]
        lane_tokens = x_decoder[:, -Nl:]
        # take encoder-ed tokens back to
        # repeat 用于沿指定维度重复张量的内容， batch，
        decoder_hist_token = self.history_mask_token.repeat(B, N, 1) # self.history_mask_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        hist_pred_mask = ~data["x_key_padding_mask"]
        for i, idx in enumerate(hist_keep_ids_list):
            decoder_hist_token[i, idx] = hist_tokens[i, : len(idx)] # 0:8
            hist_pred_mask[i, idx] = False
    # decoder_hist_token 前8个有actor， hist_pred_mask 有数据的位置上是false
        decoder_fut_token = self.future_mask_token.repeat(B, N, 1)
        future_pred_mask = ~data["x_key_padding_mask"]  #[1, num-actors]
        for i, idx in enumerate(fut_keep_ids_list):
            decoder_fut_token[i, idx] = fut_tokens[i, : len(idx)]
            future_pred_mask[i, idx] = False

        decoder_lane_token = self.lane_mask_token.repeat(B, M, 1)
        lane_pred_mask = ~data["lane_key_padding_mask"]
        for i, idx in enumerate(lane_ids_keep_list):
            decoder_lane_token[i, idx] = lane_tokens[i, : len(idx)]
            lane_pred_mask[i, idx] = False
    # decoder_hist_token 【1, 16, 128】from random tensor, 前8个有actor的 x_decoder信息
        x_decoder = torch.cat( # 是在x_decoder中，根据masked留下的actor/lane的number， 16+16+53 【1, 85, 128】此处，是把 masked——decoder + random tensor size of the other half of the masked features
            [decoder_hist_token, decoder_fut_token, decoder_lane_token], dim=1
        )
        x_decoder = x_decoder + self.decoder_pos_embed(pos_feat) # pos_feat：【1, 85, 4】-》【1, 85, 128】
        decoder_key_padding_mask = torch.cat( #[1,85]
            [
                data["x_key_padding_mask"],
                future_padding_mask.all(-1),
                data["lane_key_padding_mask"],
            ],
            dim=1,
        )

        for blk in self.decoder_blocks:
            x_decoder = blk(x_decoder, key_padding_mask=decoder_key_padding_mask)

        x_decoder = self.decoder_norm(x_decoder)
        hist_token = x_decoder[:, :N].reshape(-1, self.embed_dim)
        future_token = x_decoder[:, N: 2 * N].reshape(-1, self.embed_dim)
        lane_token = x_decoder[:, -M:]

        # 确保 L(waypoints number of lane) 不超过最大值35
        assert L <= self.max_lane_pred_dim
        # lane pred loss
        lane_pred = self.lane_pred(lane_token)
        lane_pred = lane_pred[:, :, :L * 2]
        lane_pred = lane_pred.view(B, M, L, 2)
        lane_reg_mask = ~lane_padding_mask
        lane_reg_mask[~lane_pred_mask] = False
        lane_pred_loss = F.mse_loss(
            lane_pred[lane_reg_mask], lane_normalized[lane_reg_mask]
        )

        # hist pred loss
        x_hat = self.history_pred(hist_token).view(-1, 10, 2) # [16, 50, 2 ]
        x = (data["x_positions"] - data["x_centers"].unsqueeze(-2)).view(-1, 10, 2)  # x[16, 50, 2]
        x_reg_mask = ~data["x_padding_mask"][:, :, :10] # {data["x_padding_mask"] [1, 16, 110 ]actor在每个时间步上有信息的位置为False,相反为True.} 取50, 取反 [1, 16, 50]
        x_reg_mask[~hist_pred_mask] = False
        x_reg_mask = x_reg_mask.view(-1, 10)
        hist_loss = F.l1_loss(x_hat[x_reg_mask], x[x_reg_mask])

        # future pred loss
        y_hat = self.future_pred(future_token).view(-1, 30, 2)  # B*N, 120
        y = data["y"].view(-1, 30, 2)
        reg_mask = ~data["x_padding_mask"][:, :, 10:]
        reg_mask[~future_pred_mask] = False
        reg_mask = reg_mask.view(-1, 30)
        future_loss = F.l1_loss(y_hat[reg_mask], y[reg_mask])


        loss = (
                self.loss_weight[0] * future_loss
                + self.loss_weight[1] * hist_loss
                + self.loss_weight[2] * lane_pred_loss
        )

        out = {
            "loss": loss,
            "hist_loss": hist_loss.item(),
            "future_loss": future_loss.item(),
            "lane_pred_loss": lane_pred_loss.item(),
        }

        if not self.training:
            out["x_hat"] = x_hat.view(B, N, 10, 2)
            out["y_hat"] = y_hat.view(1, B, N, 30, 2)
            out["lane_hat"] = lane_pred.view(B, M, L, 2)
            out["lane_keep_ids"] = lane_ids_keep_list
            out["hist_keep_ids"] = hist_keep_ids_list
            out["fut_keep_ids"] = fut_keep_ids_list

        return out
