"""Heterogeneous graph neural encoder (specification sections 12-14).

Each round performs three sequential updates, always with the freshest
embeddings:

1. factory update   h_f' = ELU( MLP_F [ h_f || AttnPool_{m in f}(h_m) ] )
2. machine update   h_m' = ELU( MLP_M [ h_m || h_{f(m)}' || AttnPool_{o~m}(h_o) ] )
   (the operation->machine attention uses operation-machine edge features)
3. operation update h_o' = ELU( sum_r eta_r(x_o^(r)) ) over five channels:
   self / predecessor / successor / machine (attention w/ edge features) /
   factory (assigned factory embedding; zeros for unassigned).

Predecessor/successor each have at most one neighbour, so those channels use a
masked gather instead of attention.  After ``hgnn_layers`` rounds, masked mean
pooling over factories / machines / operations (all fixed-grid nodes, virtual
missing-stage nodes included, padding excluded) forms the global representation

    h_D = [ h_F_global || h_M_global || h_O_global ]   (dim = 3 * embedding_dim)

Adaptation note: specification section 12's generic formula includes an output
nonlinearity sigma(.) and the ELU placements above follow section 13 verbatim;
both use this project's ELU convention (documented in the README).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .attention import GraphAttention, MLP
from .builder import GraphBatch


@dataclass
class GraphEmbeddings:
    h_factory: torch.Tensor     # [B, F, d]
    h_machine: torch.Tensor     # [B, M, d]
    h_operation: torch.Tensor   # [B, O, d]
    h_global: torch.Tensor      # [B, 3d]


def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the node axis, ignoring masked rows (zeros if none)."""
    weights = mask.to(h.dtype).unsqueeze(-1)          # [B, N, 1]
    total = (h * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return total / count


def _gather_masked(h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather rows by ``index`` [B, U] (-1 -> zero row)."""
    valid = (index >= 0).unsqueeze(-1)
    safe = index.clamp(min=0)
    gathered = h.gather(1, safe.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
    return gathered * valid.to(gathered.dtype)


class HGNNLayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int = 2):
        super().__init__()
        # stage 1: factories <- their machines
        self.factory_attn = GraphAttention(dim, edge_dim=None, use_self=False)
        self.factory_mlp = MLP(2 * dim, dim, dim, hidden_layers=1)
        # stage 2: machines <- assigned factory (updated) + related operations
        self.machine_attn = GraphAttention(dim, edge_dim=edge_dim, use_self=False)
        self.machine_mlp = MLP(3 * dim, dim, dim, hidden_layers=1)
        # stage 3: operations, five channels
        self.op_self = MLP(dim, dim, dim, hidden_layers=1)
        self.op_predecessor = MLP(dim, dim, dim, hidden_layers=1)
        self.op_successor = MLP(dim, dim, dim, hidden_layers=1)
        self.op_machine_attn = GraphAttention(dim, edge_dim=edge_dim, use_self=False)
        self.op_machine = MLP(dim, dim, dim, hidden_layers=1)
        self.op_factory = MLP(dim, dim, dim, hidden_layers=1)

    def forward(
        self,
        graph: GraphBatch,
        h_factory: torch.Tensor,
        h_machine: torch.Tensor,
        h_operation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---- stage 1: factory nodes ---------------------------------------
        pool_machines = self.factory_attn(
            h_factory,
            h_machine,
            graph.factory_machine_mask,
            edge_features=None,
            query_mask=graph.factory_mask,
        )
        h_factory = F.elu(
            self.factory_mlp(torch.cat([h_factory, pool_machines], dim=-1))
        )
        h_factory = h_factory * graph.factory_mask.unsqueeze(-1).to(h_factory.dtype)

        # ---- stage 2: machine nodes ---------------------------------------
        op_machine_mask_mo = graph.operation_machine_mask.transpose(1, 2)  # [B,M,O]
        edges_mo = graph.om_edge_features().transpose(1, 2)                # [B,M,O,2]
        pool_ops = self.machine_attn(
            h_machine,
            h_operation,
            op_machine_mask_mo,
            edge_features=edges_mo,
            query_mask=graph.machine_mask,
        )
        h_factory_of_machine = _gather_masked(h_factory, graph.machine_factory_index)
        h_machine = F.elu(
            self.machine_mlp(
                torch.cat([h_machine, h_factory_of_machine, pool_ops], dim=-1)
            )
        )
        h_machine = h_machine * graph.machine_mask.unsqueeze(-1).to(h_machine.dtype)

        # ---- stage 3: operation nodes -------------------------------------
        pool_machines_for_ops = self.op_machine_attn(
            h_operation,
            h_machine,
            graph.operation_machine_mask,
            edge_features=graph.om_edge_features(),
            query_mask=graph.operation_mask,
        )
        msg_self = self.op_self(h_operation)
        msg_pred = self.op_predecessor(
            _gather_masked(h_operation, graph.predecessor_index)
        )
        msg_succ = self.op_successor(
            _gather_masked(h_operation, graph.successor_index)
        )
        msg_machine = self.op_machine(pool_machines_for_ops)
        msg_factory = self.op_factory(
            _gather_masked(h_factory, graph.op_factory_index)
        )
        h_operation = F.elu(msg_self + msg_pred + msg_succ + msg_machine + msg_factory)
        h_operation = h_operation * graph.operation_mask.unsqueeze(-1).to(h_operation.dtype)
        return h_factory, h_machine, h_operation


class HGNNEncoder(nn.Module):
    def __init__(
        self,
        factory_feature_dim: int = 2,
        machine_feature_dim: int = 4,
        operation_feature_dim: int = 6,
        edge_dim: int = 2,
        embedding_dim: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        dim = int(embedding_dim)
        self.factory_encoder = MLP(factory_feature_dim, dim, dim, hidden_layers=1)
        self.machine_encoder = MLP(machine_feature_dim, dim, dim, hidden_layers=1)
        self.operation_encoder = MLP(operation_feature_dim, dim, dim, hidden_layers=1)
        self.layers = nn.ModuleList(
            [HGNNLayer(dim, edge_dim=edge_dim) for _ in range(int(num_layers))]
        )
        self.dim = dim

    def forward(self, graph: GraphBatch) -> GraphEmbeddings:
        h_factory = self.factory_encoder(graph.factory_features)
        h_machine = self.machine_encoder(graph.machine_features)
        h_operation = self.operation_encoder(graph.operation_features)

        for layer in self.layers:
            h_factory, h_machine, h_operation = layer(
                graph, h_factory, h_machine, h_operation
            )

        h_f_global = _masked_mean(h_factory, graph.factory_mask)
        h_m_global = _masked_mean(h_machine, graph.machine_mask)
        h_o_global = _masked_mean(h_operation, graph.operation_mask)
        h_global = torch.cat([h_f_global, h_m_global, h_o_global], dim=-1)
        return GraphEmbeddings(
            h_factory=h_factory,
            h_machine=h_machine,
            h_operation=h_operation,
            h_global=h_global,
        )
