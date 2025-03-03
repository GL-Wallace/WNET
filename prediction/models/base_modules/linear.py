# Copyright (c) Carizon. All rights reserved.

import math
from typing import List, Optional

import torch
import torch.nn as nn


class MLPLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        """Initialize.

        Args:
            input_dim (int): input dimension.
            hidden_dim (int): hidden dimension.
            output_dim (int): output dimension.
        """
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (tensor): input data

        Returns:
            x (tensor): output data
        """
        return self.mlp(x)


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
