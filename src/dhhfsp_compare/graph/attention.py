"""Edge-featured masked graph attention (specification section 12).

    e_uv   = LeakyReLU(a . [W_q h_u || W_k h_v || W_e xi_uv])
    alpha  = softmax_v(e_uv)          (over valid neighbours only)
    out_u  = W_s h_u + sum_v alpha_uv W_v h_v

Design notes
------------
* Neighbours are addressed by a dense ``[B, U, V]`` boolean mask.  U / V are
  the (padded) numbers of query / key nodes of one relation, which is small in
  this domain (3 factories, 45 machines, at most a few hundred grid
  operations), so dense masking is simpler and faster than index gathers.
* Masked entries get score ``-1e30`` before the softmax (never ``-inf``), and
  the softmax output is multiplied by the mask again.  Nodes with an empty
  neighbour set therefore aggregate exactly zero instead of NaN.
* Padded (batch) nodes are excluded through ``key_mask`` and zeroed through
  ``query_mask``, so single-sample and batched inference agree.
* No activation is applied to the output; the HGNN layers apply ELU at the
  points prescribed by specification section 13.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        edge_dim: int | None = None,
        use_self: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_s = nn.Linear(dim, dim, bias=False) if use_self else None
        self.w_e = nn.Linear(edge_dim, dim, bias=False) if edge_dim else None
        # attention vector a = [a_q || a_k || a_e], one slice per term
        self.a_q = nn.Parameter(torch.zeros(dim))
        self.a_k = nn.Parameter(torch.zeros(dim))
        self.a_e = nn.Parameter(torch.zeros(dim)) if edge_dim else None
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(
        self,
        query: torch.Tensor,            # [B, U, d]
        key: torch.Tensor,              # [B, V, d]
        key_mask: torch.Tensor,         # [B, U, V] bool
        edge_features: torch.Tensor | None = None,   # [B, U, V, de]
        query_mask: torch.Tensor | None = None,      # [B, U] bool
    ) -> torch.Tensor:                  # [B, U, d]
        q = self.w_q(query)                                # [B, U, d]
        k = self.w_k(key)                                  # [B, V, d]
        v = self.w_v(key)                                  # [B, V, d]

        score = torch.einsum("bud,bvd->buv", q, k)
        score = score + torch.einsum("bud,d->bu", q, self.a_q).unsqueeze(2)
        score = score + torch.einsum("bvd,d->bv", k, self.a_k).unsqueeze(1)
        if self.w_e is not None and self.a_e is not None and edge_features is not None:
            # a_e . (W_e xi) = (W_e^T a_e) . xi with W_e bias-free: this keeps
            # the edge term at [B, U, V] instead of materialising [B, U, V, d].
            c = self.w_e.weight.t() @ self.a_e             # [edge_dim]
            score = score + torch.einsum("buvd,d->buv", edge_features, c)
        score = self.leaky_relu(score)

        key_mask = key_mask.bool()
        score = score.masked_fill(~key_mask, -1e30)
        attn = torch.softmax(score, dim=-1) * key_mask.to(score.dtype)
        out = torch.einsum("buv,bvd->bud", attn, v)
        if self.w_s is not None:
            out = out + self.w_s(query)
        if query_mask is not None:
            out = out * query_mask.to(out.dtype).unsqueeze(-1)
        return out


class MLP(nn.Module):
    """Small helper: Linear -> ELU -> ... -> Linear (hidden layers configurable)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, hidden_layers: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.ELU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
