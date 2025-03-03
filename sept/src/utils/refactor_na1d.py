#################################################################################################
# Copyright (c)  .
#################################################################################################
import math
import torch
from torch import nn, Tensor
from typing import Optional
import torch.nn.functional as F


class RefactorNeighborhoodAttention1D(nn.Module):
    """
    Refactor Neighborhood Attention 1D Module

    """

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.ln = torch.nn.LayerNorm(dim)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.W_rpe_k = nn.Linear(self.dim, self.dim)
        self.W_rpe_v = nn.Linear(self.dim, self.dim)

    def forward(
            self,
            Q: Tensor,
            K: Tensor,
            V: Tensor,
            rpe: Optional[torch.Tensor] = None,
            window_size: Optional[int] = 15,
            attention_mask: Optional[Tensor] = None,
            # knn_idxs: Optional[torch.Tensor] = None,\
    ) -> Tensor:

        if Q.dim() != 3 or K.dim() != 3 or V.dim() != 3:
            raise ValueError(
                f"NeighborhoodAttention1D expected a rank-3 input tensor; got {Q.dim(), K.dim(), V.dim()}."
            )
        B, L, C = Q.shape
        _, target_len, _ = K.shape
        q = self.w_q(Q).reshape(B, self.num_heads, -1, self.head_dim)
        k = self.w_k(K).reshape(B, self.num_heads, -1, self.head_dim)
        v = self.w_v(V).reshape(B, self.num_heads, -1, self.head_dim)

        if rpe is not None:
            rpe = self.ln(rpe)
            rpe_k = self.W_rpe_k(rpe).reshape(B, L, L, self.num_heads, -1).permute(0, 3, 1, 2, 4)
            rpe_v = self.W_rpe_v(rpe).reshape(B, L, L, self.num_heads, -1).permute(0, 3, 1, 2, 4)
            q1 = (
                q[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            k1 = (
                k[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            v1 = (
                v[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            k1 += rpe_k
            v1 += rpe_v

            q1 = q1.reshape(B, self.num_heads, L * L, -1)
            k1 = k1.reshape(B, self.num_heads, L * L, -1)  # [B, H, L*L, D]
            v1 = v1.reshape(B, self.num_heads, L * L, -1)  # [B, H, L*L, D]

            pad_size = window_size // 2
            Q_pad = F.pad(q1, (0, 0, pad_size, pad_size))
            K_pad = F.pad(k1, (0, 0, pad_size, pad_size))
            V_pad = F.pad(v1, (0, 0, pad_size, pad_size))

            # Create sliding windows
            Q_windows = Q_pad.unfold(2, window_size, 1).permute(0, 1, 2, 4, 3)
            K_windows = K_pad.unfold(2, window_size, 1).permute(0, 1, 2, 4, 3)
            V_windows = V_pad.unfold(2, window_size, 1).permute(0, 1, 2, 4, 3)

            # Calculate attention within local windows
            attn_mat = (Q_windows * K_windows).sum(dim=-1) / self.scale
            attn_mat = F.softmax(attn_mat, dim=-1)
            attn_mat = self.attn_drop(attn_mat)

            # Weighted sum of values
            qv = ((attn_mat[:, :, :, :, None] * V_windows).sum(dim=-2)
                  .reshape(B, self.num_heads, L, L, -1)
                  .sum(-2)
                  .permute(0, 2, 1, 3)
                  .reshape(B, L, -1))

        else:

            attention_weight = (
                    q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
            )

            if attention_mask is not None:
                attention_weight = attention_weight.masked_fill(
                    attention_mask == 0, float("-1e20")
                )

            attention_weight = torch.softmax(attention_weight, dim=3)

            attention_weight = self.attn_drop(attention_weight)
            qv = (attention_weight @ v).transpose(1, 2).contiguous().view(B, L, -1)

        output = self.proj(qv)
        output = self.proj_drop(output)
        return output


class RefactorNeighborhoodAttention2D(nn.Module):

    def __init__(self, embed_dim, num_heads, window_size, dropout=0.0):
        super(RefactorNeighborhoodAttention2D, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, T, C = x.size()
        q = self.q_proj(x).view(B, N, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        k = self.k_proj(x).view(B, N, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = self.v_proj(x).view(B, N, T, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # Pad k and v for local attention
        pad_size = self.window_size // 2
        k = F.pad(k, (0, 0, pad_size, pad_size, pad_size, pad_size))
        v = F.pad(v, (0, 0, pad_size, pad_size, pad_size, pad_size))

        attn_output = []

        attn_output = torch.stack(attn_output, dim=2).view(B, self.num_heads, H, W, self.head_dim)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).contiguous().view(B, H, W, C)


        return self.out_proj(attn_output)

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, nums_key_value_head, dropout=0.0):
        super(GroupQueryAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        assert num_heads % nums_key_value_head == 0, "nums_key_value_head must be devided by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.nums_key_value_head = num_heads // 2
        self.head_dim = embed_dim // num_heads
        self.ln = torch.nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(embed_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, nums_key_value_head * self.head_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.W_rpe_k = nn.Linear(self.embed_dim, nums_key_value_head * self.head_dim)
        self.W_rpe_v = nn.Linear(self.embed_dim, nums_key_value_head * self.head_dim)

    def forward(self, Q, K, V, mask=None, rpe: Optional[torch.Tensor] = None):
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(V)
        B, L, _ = q.size()

        q = q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, -1, self.nums_key_value_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.nums_key_value_head, self.head_dim).permute(0, 2, 1, 3)

        if rpe is not None:
            rpe = self.ln(rpe)
            rpe_k = self.W_rpe_k(rpe).reshape(B, L, L, self.nums_key_value_head, -1).permute(0, 3, 1, 2, 4)
            rpe_v = self.W_rpe_v(rpe).reshape(B, L, L, self.nums_key_value_head, -1).permute(0, 3, 1, 2, 4)
            q1 = (
                q[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            k1 = (
                k[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            v1 = (
                v[:, :, :, None, :]
                .repeat(1, 1, 1, L, 1)
            )
            k1 += rpe_k
            v1 += rpe_v

            q1 = q1.reshape(B, self.num_heads, L * L, -1)
            k1 = k1.reshape(B, self.nums_key_value_head, L * L, -1)  # [B, H, L*L, D]
            v1 = v1.reshape(B, self.nums_key_value_head, L * L, -1)  # [B, H, L*L, D]
            k1 = k1.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=1)
            v1 = v1.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=1)
            # Calculate attention within local windows
            attn_mat = (q1 * k1).sum(dim=-1) / math.sqrt(self.head_dim)
            attn_mat = F.softmax(attn_mat, dim=-1)
            attn_mat = self.attn_drop(attn_mat)
            # Weighted sum of values
            qv = ((attn_mat[:, :, :, None] * v1)
                  .reshape(B, self.num_heads, L, L, -1)
                  .sum(-2)
                  .permute(0, 2, 1, 3)
                  .reshape(B, L, -1))
        else:

            k = k.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=1)
            v = v.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=1)

            attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            qv = (attn @ v).transpose(1, 2).contiguous().view(B, L, -1)

        output = self.proj_drop(self.proj(qv))
        return output