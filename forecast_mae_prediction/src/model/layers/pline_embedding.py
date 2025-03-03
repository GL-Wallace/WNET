import torch
import torch.nn as nn


class LaneEmbeddingLayer(nn.Module):
    def __init__(self, feat_channel, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(feat_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, self.encoder_channel, 1),
        )

    def forward(self, x):
        bs, n, _ = x.shape # bs：53, N: 20, _d:3
        feature = x.transpose(2, 1).float()
        feature = self.first_conv(feature)  # x_input(53, 3, 20) output: Bs 256 20
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )
        feature = self.second_conv(feature)
        # feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B c 【53, 128】
        return feature
