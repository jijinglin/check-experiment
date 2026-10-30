"""HGNN-PPO actor-critic: joint action scoring + single critic (sections 15-17).

Both action kinds are scored by separate MLPs but share ONE actor distribution:
``policy_value`` returns the concatenated masked logits ``[B, A1 + A2]`` (the
caller applies a single log-softmax) and a single scalar value per state.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..graph.attention import MLP
from ..graph.builder import GraphBatch
from ..graph.hgnn import GraphEmbeddings, HGNNEncoder


class BaoActorCritic(nn.Module):
    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        dim = int(model_cfg.get("embedding_dim", 8))
        self.dim = dim
        self.encoder = HGNNEncoder(
            embedding_dim=dim,
            num_layers=int(model_cfg.get("hgnn_layers", 2)),
        )
        actor_hidden = int(model_cfg.get("actor_hidden_dim", 128))
        critic_hidden = int(model_cfg.get("critic_hidden_dim", 128))
        # inputs: [h_entity (d) || h_partner (d) || h_global (3d) || 2 extra features]
        self.assign_scorer = MLP(5 * dim + 2, actor_hidden, 1, hidden_layers=2)
        self.schedule_scorer = MLP(5 * dim + 2, actor_hidden, 1, hidden_layers=2)
        self.critic = MLP(3 * dim, critic_hidden, 1, hidden_layers=2)

    # -------------------------------------------------------------- encoding
    def encode_graph(self, graph_batch: GraphBatch) -> GraphEmbeddings:
        return self.encoder(graph_batch)

    def _job_embeddings(
        self, embeddings: GraphEmbeddings, graph_batch: GraphBatch
    ) -> torch.Tensor:
        """Aggregate each job's real operation embeddings -> [B, J, d]."""
        B, O, d = embeddings.h_operation.shape
        K = graph_batch.n_stages
        J = O // K
        h = embeddings.h_operation.view(B, J, K, d)
        real = graph_batch.real_op_mask.view(B, J, K).to(h.dtype).unsqueeze(-1)
        counts = real.sum(dim=2).clamp(min=1.0)
        return (h * real).sum(dim=2) / counts

    # -------------------------------------------------------------- scorers
    def score_assignment_actions(
        self, embeddings: GraphEmbeddings, graph_batch: GraphBatch
    ) -> torch.Tensor:
        """Job->factory logits [B, A1]."""
        B, A1 = graph_batch.assign_job.shape
        d = self.dim
        h_jobs = self._job_embeddings(embeddings, graph_batch)      # [B, J, d]
        job = graph_batch.assign_job.clamp(min=0)
        factory = graph_batch.assign_factory.clamp(min=0)
        h_job = h_jobs.gather(1, job.unsqueeze(-1).expand(-1, -1, d))
        h_fac = embeddings.h_factory.gather(
            1, factory.unsqueeze(-1).expand(-1, -1, d)
        )
        h_global = embeddings.h_global.unsqueeze(1).expand(-1, A1, -1)
        features = torch.cat(
            [h_job, h_fac, h_global, graph_batch.assign_extra], dim=-1
        )
        return self.assign_scorer(features).squeeze(-1)

    def score_scheduling_actions(
        self, embeddings: GraphEmbeddings, graph_batch: GraphBatch
    ) -> torch.Tensor:
        """Operation->machine logits [B, A2]."""
        B, A2 = graph_batch.schedule_job.shape
        d = self.dim
        grid = graph_batch.schedule_op_grid.clamp(min=0)
        machine = graph_batch.schedule_machine.clamp(min=0)
        h_op = embeddings.h_operation.gather(
            1, grid.unsqueeze(-1).expand(-1, -1, d)
        )
        h_mach = embeddings.h_machine.gather(
            1, machine.unsqueeze(-1).expand(-1, -1, d)
        )
        h_global = embeddings.h_global.unsqueeze(1).expand(-1, A2, -1)
        features = torch.cat(
            [h_op, h_mach, h_global, graph_batch.schedule_extra], dim=-1
        )
        return self.schedule_scorer(features).squeeze(-1)

    # ---------------------------------------------------------------- joint
    def policy_value(
        self, graph_batch: GraphBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint masked logits [B, A1 + A2] and state value [B]."""
        embeddings = self.encode_graph(graph_batch)
        assign_logits = self.score_assignment_actions(embeddings, graph_batch)
        schedule_logits = self.score_scheduling_actions(embeddings, graph_batch)
        logits = torch.cat([assign_logits, schedule_logits], dim=-1)
        logits = logits.masked_fill(
            ~graph_batch.action_mask, float("-inf")
        )
        value = self.critic(embeddings.h_global).squeeze(-1)
        return logits, value
