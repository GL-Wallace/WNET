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
        # torch.max(input, dim, keepdim)：返回沿指定维度的最大值及其索引;
        # 返回一个包含两个元素的元组，第一个元素是最大值，第二个元素是最大值的索引。我们只需要最大值，所以取[0]。
        # keepdim=True: 原本的 dim=2 维度（长度为 3）被保留，但长度变为 1。
        # 这样就确保了输出张量 feature_global 的维度与输入张量 feature 的维度一致，只是dim=2的长度变成了 1。保持输出张量的维度; [53, 256, 1]
        feature = x.transpose(2, 1).float()

        feature = self.first_conv(feature)  # x_input(53, 3, 20) output: Bs 256 20
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # B 256 1
        # 问题1： 为什么要取最大值？ 从3到256 取最大值到底代表着什么？  为什么要拼成512?
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # B 512 20 = B 256 20 + B 256 20；
        feature = self.second_conv(feature)  # B c n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # B c 【53, 128】
        return feature_global
