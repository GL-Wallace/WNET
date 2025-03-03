import torch
import torch.nn as nn
import torch.nn.functional as F
from natten import NeighborhoodAttention1D
from timm.models.layers import DropPath
from forecast_mae_prediction.src.utils.build_rpes import BuildRPES


class AgentEmbeddingLayer(nn.Module):
    def __init__(
            self,
            in_chans=4,
            embed_dim=32,
            mlp_ratio=3,
            kernel_size=[3, 3, 3],
            depths=[2, 2, 2],
            num_heads=[2, 4, 8],
            out_indices=[0, 1, 2],
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.embed = ConvTokenizer(in_chans, embed_dim)
        self.num_levels = len(depths)
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.out_indices = out_indices
        self.build_rpes= BuildRPES(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                # ** > *  i[0, 1, 2]
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
                dilations=None,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]): sum(depths[: i + 1])],
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

        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)

    def forward(self, x):
        """x: [B,D, N, T]"""
        rpe_a2a = self.build_rpes.build_a2a_rpes(x)
        x = x[:, :4, :, :]
        x = self.embed(x)  # [B, N, T, emb_dim]

        out = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            # getattr 用于动态地获取对象的属性或方法, layerNorm
            # 对这些特征进行归一化，然后将处理后的特征存储在一个列表中
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                out.append(x_out.permute(0, 2, 1).contiguous())

        laterals = [
            # laterals[]
            # idx 0: conv1d(32, 128，3, 1) [16, 32, 50] _> [16, 128, 50]
            # idx 1: conv1d(64, 128，3, 1) [16, 64, 25] -> [16, 128, 25]
            # idx 2: conv1d(128, 128，3, 1) [16, 128, 13] -> [16,128, 13]
            lateral_conv(out[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(out) - 1, 0, -1):
            # F.interpolate() 上采样，将L[2], 上采样2倍，+ L[1] =》update L[1]
            # F.interpolate() 上采样，将L[1], 上采样2倍，+ L[0] =》update L[0]
            # 这个逻辑是： 将低分辨率信息上采样，最终把所有信息更新到最高分辨率特征中
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                scale_factor=(laterals[i - 1].shape[-1] / laterals[i].shape[-1]),
                mode="linear",
                align_corners=False,
            )
        # out = fpn_conv1d(128, 128, 3, 1, 1) [16, 128, 50] _> [16, 128, 50]
        out = self.fpn_conv(laterals[0])
        # 将返回一个形状为 [16, 128] 的张量，其中包含了 out 在第三个维度上的最后一个元素。最后一个时间步的输出作为整个序列的特征表示
        return out[:, :, -1]


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=4, embed_dim=32, norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # [B, N, T, D 32]
        if self.norm is not None:
            x = self.norm(x)
        return x


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
            kernel_size=7,
            dilation=None,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention1D(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)  # [16, 50, 32] - [16,50,32] didn't change
        x = self.drop_path(x)
        x = shortcut + x
        x = self.norm2(x)
        x = self.mlp(x)  # [16, 50, 32]
        x = shortcut + self.drop_path(x)
        return x


class NATBlock(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads,
            kernel_size,
            dilations=None,
            downsample=True,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
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
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x
