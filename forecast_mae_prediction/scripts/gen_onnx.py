import os
import torch
import torch.nn as nn
import onnx
from forecast_mae_prediction.src.model.layers.agent_embedding import AgentEmbeddingLayer
from forecast_mae_prediction.src.model.layers.lane_embedding import LaneEmbeddingLayer
from forecast_mae_prediction.src.model.layers.multimodal_decoder import MultimodalDecoder
from forecast_mae_prediction.src.model.layers.transformer_blocks import Block
from forecast_mae_prediction.src.datamodule.lmdb_dataset import collate_fn


class ForecastEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: object = 128,
            encoder_depth: object = 4,
            num_heads: object = 8,
            mlp_ratio: object = 4.0,
            qkv_bias: object = False,
            drop_path: object = 0.2,
            future_steps: int = 30,
            k: int = 3,
    ) -> None:
        super(ForecastEncoder, self).__init__()
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

    def forward(self, x_padding_mask, x_key_padding_mask, x, x_velocity_diff, lane_padding_mask, lane_positions,
                lane_centers, x_angles, lane_angles, lane_key_padding_mask, x_centers, x_attr):
        hist_padding_mask = x_padding_mask[:, :, :10]
        hist_key_padding_mask = x_key_padding_mask
        hist_feat = torch.cat(
            [
                x,
                x_velocity_diff[..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)  # [B, N, T, D]
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])
        lane_padding_mask = lane_padding_mask
        lane_normalized = lane_positions - lane_centers.unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([x_centers, lane_centers], dim=1)  # [1,69,2]
        angles = torch.cat([x_angles[:, :, 9], lane_angles], dim=1)  # [1, 69]
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [1,69,2]
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)  # [1, 69, 4]
        pos_embed = self.pos_embed(pos_feat)  # [1, 69, 128]

        actor_type_embed = self.actor_type_embed[x_attr[..., 0].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [x_key_padding_mask, lane_key_padding_mask], dim=1
        )

        x_encoder = x_encoder + pos_embed
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        x_agent = x_encoder[:, 0]
        y_hat, pi = self.decoder(x_agent)

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, -1, 30, 2)

        return y_hat, pi, y_hat_others

class ForecastDecoder(nn.Module):
    def __init__(self, ):
        super(ForecastDecoder, self).__init__()

    def forward(self, y_hat, pi, y_hat_others):
        predictions = y_hat
        pi = pi
        return predictions, pi

class ForecastMAEONNX(nn.Module):
    def __init__(self, ):
        super(ForecastMAEONNX, self).__init__()

        self.encoder = ForecastEncoder(
            embed_dim=128,
            encoder_depth=4,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_path=0.2,
            future_steps=30,
            k=3 )
        self.decoder = ForecastDecoder()

    def forward(self, x_padding_mask, x_key_padding_mask, x, x_velocity_diff, lane_padding_mask, lane_positions,
                lane_centers, x_angles, lane_angles, lane_key_padding_mask, x_centers, x_attr):
        y_hat, pi, y_others = self.encoder(x_padding_mask, x_key_padding_mask, x, x_velocity_diff,
                                           lane_padding_mask, lane_positions,
                                           lane_centers, x_angles, lane_angles, lane_key_padding_mask, x_centers,
                                           x_attr)
        prediction, prob = self.decoder(y_hat, pi, y_others)
        return prediction, prob


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_input = "/home/user/Documents/pt_files/0/1722513825.834000_60.pt"
    ckpt_path = '/home/user/Downloads/epoch=86.ckpt'
    onnx_dir = os.path.join("/home/user/Projects/pnp_research/forecast_mae_prediction/outputs", "onnx")

    ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state_dict = {
        k.replace("net.", "encoder."): v for k, v in ckpt.items() if k.startswith("net.")
    }
    model = ForecastMAEONNX()
    model.load_state_dict(state_dict=state_dict, strict=False)
    model.eval().to(device)

    with open(sample_input, "rb") as f:
        sample_data = torch.load(f)
    data = collate_fn([sample_data])

    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)

    x_padding_mask = data["x_padding_mask"]
    x_key_padding_mask = data["x_key_padding_mask"]
    x = data["x"]
    x_velocity_diff = data["x_velocity_diff"]
    lane_padding_mask = data["lane_padding_mask"]
    lane_positions = data["lane_positions"]
    lane_centers = data["lane_centers"]
    x_angles = data["x_angles"]
    lane_angles = data["lane_angles"]
    lane_key_padding_mask = data["lane_key_padding_mask"]
    x_centers = data["x_centers"]
    x_attr = data["x_attr"]
    lane_centers = data["lane_centers"]

    torch.onnx.export(
        model,
        args=(x_padding_mask, x_key_padding_mask, x, x_velocity_diff, lane_padding_mask, lane_positions,
              lane_centers, x_angles, lane_angles, lane_key_padding_mask, x_centers, x_attr),
        f=os.path.join(onnx_dir, "forecast_mae.onnx"),
        export_params=True, # 模型中是否存储模型权重
        opset_version=14,  # 指定所用的ONNX操作集的版本
        do_constant_folding=True,  # 是否使用“常量折叠”优化。常量折叠将使用一些算好的常量来优化一些输入全为常量的节点
        training=torch.onnx.TrainingMode.EVAL,
        input_names=["x_padding_mask", "x_key_padding_mask", "x", "x_velocity_diff", "lane_padding_mask",
                     "lane_positions",
                     "lane_centers", "x_angles", "lane_angles", "lane_key_padding_mask", "x_centers", "x_attr"],
        output_names=["prediction", "prob"],
    )

# print("ONNX 模型导出成功，文件已保存至:", os.path.join(onnx_dir, "model.onnx"))