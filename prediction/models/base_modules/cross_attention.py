# Copyright (c) Carizon. All rights reserved.

import math
from typing import Optional

import torch
import torch.nn as nn


class RPEMHALayer(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_head: int,
        dropout=0.1,
    ) -> None:
        """General multi-head attention module.

        This module supports relative
        positional encoding and knn-based attention.

        Args:
            d_model (int): the dimension of hidden features.
            n_head (int): the number of heads.
            L_q (int): the query sequence length.
            L_k (int): the k/v sequence length.
            dropout (float, optional): the dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = math.sqrt(self.d_head)

        self.ln = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        self.W_q = nn.Linear(self.d_input, self.d_model)
        self.W_k = nn.Linear(self.d_input, self.d_model)
        self.W_v = nn.Linear(self.d_input, self.d_model)
        self.W_rpe_k = nn.Linear(self.d_input, self.d_model, bias=False)
        self.W_rpe_v = nn.Linear(self.d_input, self.d_model)
        self.attn_out = nn.Linear(self.d_head * self.n_head, self.d_input)
        self.mlp_out = nn.Sequential(
            nn.Linear(self.d_input, self.d_input),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.d_input, d_input),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rpe: Optional[torch.Tensor] = None,
        knn_idxs: Optional[torch.Tensor] = None,
        invalid_mask: Optional[torch.Tensor] = None,
    ):
        """Forward.

        Args:
            q (torch.Tensor): [B, L_q, D_in]
            k (torch.Tensor): [B, L_k, D_in]
            v (torch.Tensor): [B, L_k, D_in]
            rpe (Optional[torch.Tensor], optional): The pre-computed
                relative positional encodings in the shape [B, L_q, L_k, D_in],
                representing the pair-wise relation between tokens in query
                and key/value. Defaults to None.
            knn_idxs (Optional[torch.Tensor], optional): The indices of the
                query's k-nearest neighbors in KV in the shape [B, L_q, NB_k].
                Defaults to None.
            invalid_mask (Optional[torch.Tensor], optional): The mask to
                apply to the attention matrix in the shape [B, L_q, L_k].
                Usually for masking the padded sequences. A True valu in the
                mask indicates that the corresponding element should be
                ignored. Defaults to None.

        Returns:
            q_out: The output query tensor in the shape [B, L_q, D_in].
        """
        B, L_q, L_k, H, D = (
            q.size(0),
            q.size(1),
            k.size(1),
            self.n_head,
            self.d_head,
        )

        q = self.ln(q)
        k = self.ln(k)
        v = self.ln(v)

        if rpe is not None:
            rpe = self.ln(rpe)
            q1 = (
                self.W_q(q).reshape(B, L_q, H, D).permute(0, 2, 1, 3)
            )  # [B, H, L_q, D]
            k1 = (
                self.W_k(k)
                .reshape(B, L_k, H, D)
                .permute(0, 2, 1, 3)[:, :, None, :, :]
                .repeat(1, 1, L_q, 1, 1)
            )  # [B, H, L_q, L_k, D]
            v1 = (
                self.W_v(v)
                .reshape(B, L_k, H, D)
                .permute(0, 2, 1, 3)[:, :, None, :, :]
                .repeat(1, 1, L_q, 1, 1)
            )  # [B, H, L_q, L_k, D]
            rpe_k = (
                self.W_rpe_k(rpe)
                .reshape(B, L_q, L_k, H, -1)
                .permute(0, 3, 1, 2, 4)
            )  # [B, H, L_q, L_k, D]
            rpe_v = (
                self.W_rpe_v(rpe)
                .reshape(B, L_q, L_k, H, -1)
                .permute(0, 3, 1, 2, 4)
            )  # [B, H, L_q, L_k, D]
            if knn_idxs is not None:
                NB_k = knn_idxs.shape[2]
                assert NB_k <= L_k
                gather_idxs = knn_idxs[:, None, :, :, None].repeat(
                    1, H, 1, 1, D
                )  # [B, H, L_q, NB_k, D]
                k1 = k1.gather(
                    dim=3, index=gather_idxs
                )  # [B, H, L_q, NB_k, D]
                v1 = v1.gather(
                    dim=3, index=gather_idxs
                )  # [B, H, L_q, NB_k, D]
                rpe_k = rpe_k.gather(
                    dim=3, index=gather_idxs
                )  # [B, H, L_q, NB_k, D]
                rpe_v = rpe_v.gather(
                    dim=3, index=gather_idxs
                )  # [B, H, L_q, NB_k, D]
                invalid_mask = invalid_mask.gather(
                    dim=2, index=knn_idxs
                )  # [B, L_q, NB_k]

            k1 = k1 + rpe_k
            v1 = v1 + rpe_v

            L_k = k1.shape[3]
            q1 = (
                q1[:, :, :, None, :]
                .repeat(1, 1, 1, L_k, 1)
                .reshape(B, H, L_q * L_k, -1)
            )  # [B, H, L_q*L_k, D]
            k1 = k1.reshape(B, H, L_q * L_k, -1)  # [B, H, L_q*L_k, D]
            v1 = v1.reshape(B, H, L_q * L_k, -1)  # [B, H, L_q*L_k, D]

            attn_mat = (q1 * k1).sum(-1) / self.scale  # [B, H, L_q*L_k]

            if invalid_mask is not None:
                MASKING_VALUE = (
                    -1e8 if attn_mat.dtype == torch.float32 else -1e4
                )
                attn_mat = attn_mat.masked_fill(
                    invalid_mask[:, None, :, :].reshape(B, 1, L_q * L_k),
                    MASKING_VALUE,
                )
            attn_mat = torch.softmax(attn_mat, dim=-1)  # [B, H, L_q*L_k]
            attn_mat = self.dropout(attn_mat)
            qv = (
                (attn_mat[:, :, :, None] * v1)
                .reshape(B, H, L_q, L_k, -1)
                .sum(-2)  # sum over L_k
                .permute(0, 2, 1, 3)
                .reshape(B, L_q, -1)
            )  # [B, L_q, H*D]
        else:
            q1 = self.W_q(q).reshape(B, L_q, H, D).permute(0, 2, 1, 3)
            k1 = self.W_k(k).reshape(B, L_k, H, D).permute(0, 2, 1, 3)
            v1 = self.W_v(v).reshape(B, L_k, H, D).permute(0, 2, 1, 3)
            attn_mat = torch.matmul(q1, k1.transpose(-1, -2)) / self.scale
            if invalid_mask is not None:
                MASKING_VALUE = (
                    -1e8 if attn_mat.dtype == torch.float32 else -1e4
                )
                attn_mat = attn_mat.masked_fill(
                    invalid_mask[:, None, :, :], MASKING_VALUE
                )
            attn_mat = self.dropout(attn_mat)
            qv = torch.matmul(
                attn_mat,  # [B, H, L_q, L_k]
                v1,  # [B, H, L_k, D]
            )  # [B, H, L_q, D]
            qv = qv.permute(0, 2, 1, 3).reshape(B, L_q, H * D)  # [B, L_q, H*D]

        qv = q + self.ln(self.attn_out(qv))  # [B, L_q, D_in]
        qv = qv + self.ln(self.mlp_out(self.ln(qv)))  # [B, L_q, D_in]
        return qv
