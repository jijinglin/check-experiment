"""Graph assembly for one scheduling state + batch collation (specification 11).

Fixed-grid conventions
----------------------
Operations live on the fixed ``[J, K]`` grid, row-major: grid index
``o = j * K + k``.  Missing (skipped) stages are *virtual* nodes of the same
grid: they keep a node slot, carry zero features and are flagged by
``real_op_mask``; they never connect to machines.  Precedence edges follow the
grid adjacency ``k-1 <-> k+1`` so chains may pass through virtual nodes while
the structure stays uniform (specification 8.2).

The specification lists an ``operation_precedence_mask [B, O, O]``; storing that
dense ``O x O`` tensor per transition would dominate memory (O up to ~400 in
the OOD splits), so the same relation is stored as two index vectors
``predecessor_index`` / ``successor_index`` (``-1`` = no predecessor/successor).
With at most one neighbour per direction this is exact, not an approximation.

The operation-machine *edge features* (processing time, actually-selected
flag) are derived lazily from ``op_base_scaled`` x ``machine_speed`` and
``op_actual_machine`` via :meth:`GraphBatch.om_edge_features`, so the stored
tensors stay compact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch

from ..core.actions import is_assign, parse_assign, parse_schedule
from ..core.simulator import ScheduleState
from .features import (
    StateFeatures,
    assignment_action_extra,
    compute_state_features,
    scheduling_action_extra,
)


@dataclass
class GraphBatch:
    """One (or a padded batch of) decision-state graph(s); batch dim first."""

    # node features
    factory_features: torch.Tensor        # [B, F, 2] float32
    machine_features: torch.Tensor        # [B, M, 4] float32
    operation_features: torch.Tensor      # [B, O, 6] float32

    # node masks
    factory_mask: torch.Tensor            # [B, F] bool
    machine_mask: torch.Tensor            # [B, M] bool
    operation_mask: torch.Tensor          # [B, O] bool (grid nodes incl. virtual)
    real_op_mask: torch.Tensor            # [B, O] bool

    # relations
    factory_machine_mask: torch.Tensor    # [B, F, M] bool
    operation_machine_mask: torch.Tensor  # [B, O, M] bool

    # precedence (grid adjacency; -1 = none)
    predecessor_index: torch.Tensor       # [B, O] int64
    successor_index: torch.Tensor         # [B, O] int64

    # per-node static lookups
    op_job_index: torch.Tensor            # [B, O] int64
    op_stage_index: torch.Tensor          # [B, O] int64
    op_factory_index: torch.Tensor        # [B, O] int64 (-1 = job unassigned)
    op_base_scaled: torch.Tensor          # [B, O] float32 (base tick / P_ref)
    op_actual_machine: torch.Tensor       # [B, O] int64 (-1 = not scheduled yet)
    machine_speed: torch.Tensor           # [B, M] float32 (raw speed factor)
    machine_stage_index: torch.Tensor     # [B, M] int64
    machine_factory_index: torch.Tensor   # [B, M] int64

    # candidate actions (assignment block then scheduling block)
    # Logit layout per sample: [ assign block (width A1, valid first n_assign) |
    # schedule block (width A2, valid first n_schedule) ].  A block keeps a
    # width of at least 1 so empty blocks stay representable.
    assign_job: torch.Tensor              # [B, A1] int64
    assign_factory: torch.Tensor          # [B, A1] int64
    assign_extra: torch.Tensor            # [B, A1, 2] float32
    schedule_job: torch.Tensor            # [B, A2] int64
    schedule_op_grid: torch.Tensor        # [B, A2] int64 (grid index o = j*K + stage)
    schedule_machine: torch.Tensor        # [B, A2] int64
    schedule_extra: torch.Tensor          # [B, A2, 2] float32
    action_mask: torch.Tensor             # [B, A1 + A2] bool
    n_assign: torch.Tensor                # [B] int64 (valid entries per sample)
    n_schedule: torch.Tensor              # [B] int64

    # python-side bookkeeping
    actions: list[list[tuple]] = field(default_factory=list)
    n_jobs: tuple[int, ...] = ()
    n_stages: int = 0

    # ------------------------------------------------------------ utilities
    @property
    def batch_size(self) -> int:
        return int(self.factory_features.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.action_mask.shape[1])

    def to(self, device: torch.device | str) -> "GraphBatch":
        out = {}
        for key, value in self.__dict__.items():
            out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        return GraphBatch(**out)

    def om_edge_features(self) -> torch.Tensor:
        """Operation-machine edge features ``[B, O, M, 2]``.

        channel 0: exact processing time / P_ref (0 for virtual ops / non-edges)
        channel 1: 1.0 if this op is actually scheduled on this machine
        """
        dur = self.op_base_scaled.unsqueeze(2) * self.machine_speed.unsqueeze(1)  # [B,O,M]
        m_ids = torch.arange(self.machine_speed.shape[1], device=dur.device)
        actual = (self.op_actual_machine.unsqueeze(2) == m_ids.view(1, 1, -1)).to(dur.dtype)
        mask = self.operation_machine_mask.to(dur.dtype)
        return torch.stack([dur * mask, actual * mask], dim=-1)

    # ------------------------------------------------- action index mapping
    def flat_index(self, is_schedule: bool, position: int, sample: int = 0) -> int:
        """Position in the joint logit vector for block ``position``.

        A transition stores ``(is_schedule, position)`` (layout-independent);
        the flat logit position depends on the collated block width A1.
        """
        return int(self.assign_job.shape[1]) + int(position) if is_schedule else int(position)

    def action_of_flat(self, flat: int, sample: int = 0) -> tuple:
        """Map a flat logit index back to the action tuple of ``sample``."""
        w1 = int(self.assign_job.shape[1])
        n1 = int(self.n_assign[sample])
        n2 = int(self.n_schedule[sample])
        if flat < w1:
            if flat >= n1:
                raise IndexError(f"flat index {flat} points at assign padding")
            action_index = flat
        else:
            position = flat - w1
            if position >= n2:
                raise IndexError(f"flat index {flat} points at schedule padding")
            action_index = n1 + position
        return self.actions[sample][action_index]


def _factory_machine_mask(tables) -> np.ndarray:
    M = tables.n_machines
    F = tables.n_factories
    mask = np.zeros((F, M), dtype=bool)
    for m in range(M):
        mask[int(tables.machine_factory[m]), m] = True
    return mask


def _operation_machine_mask(state: ScheduleState) -> np.ndarray:
    """Operation-machine potential edges (specification section 11).

    * unassigned real operation  -> machines of its stage in every eligible factory
    * assigned, unscheduled      -> machines of (assigned factory, stage)
    * scheduled                  -> machines of (assigned factory, stage); the
      actually selected machine is flagged via ``op_actual_machine``
    * virtual / missing stage    -> no machine edges
    """
    tables = state.tables
    K = tables.n_stages
    J = tables.n_jobs
    M = tables.n_machines
    mask = np.zeros((J * K, M), dtype=bool)
    mach_f = tables.machine_factory
    mach_k = tables.machine_stage
    for j in range(J):
        assigned_f = state.assigned_factory[j]
        if assigned_f is not None:
            for k in tables.job_route[j]:
                mask[j * K + k] = (mach_k == k) & (mach_f == assigned_f)
        else:
            eligible = [
                f for f in range(tables.n_factories) if tables.eligible_factory[j, f]
            ]
            in_eligible = np.isin(mach_f, eligible)
            for k in tables.job_route[j]:
                mask[j * K + k] = (mach_k == k) & in_eligible
    return mask


def build_graph(state: ScheduleState, actions: Sequence[tuple]) -> GraphBatch:
    """Build a single-state :class:`GraphBatch` (batch size 1)."""
    tables = state.tables
    K = tables.n_stages
    J = tables.n_jobs
    F = tables.n_factories
    M = tables.n_machines
    O = J * K
    sf: StateFeatures = compute_state_features(state)

    # ---- node features -----------------------------------------------------
    factory_features = torch.from_numpy(sf.factory).unsqueeze(0)
    machine_features = torch.from_numpy(sf.machine).unsqueeze(0)
    operation_features = torch.from_numpy(sf.operation).unsqueeze(0)

    factory_mask = torch.ones(1, F, dtype=torch.bool)
    machine_mask = torch.ones(1, M, dtype=torch.bool)
    operation_mask = torch.ones(1, O, dtype=torch.bool)
    real_op_mask = torch.from_numpy(sf.real_op).unsqueeze(0)

    # ---- relations ---------------------------------------------------------
    fmm = torch.from_numpy(_factory_machine_mask(tables)).unsqueeze(0)
    omm = torch.from_numpy(_operation_machine_mask(state)).unsqueeze(0)

    o_idx = np.arange(O)
    job_of = o_idx // K
    stage_of = o_idx % K
    pred = np.where(stage_of > 0, o_idx - 1, -1)
    succ = np.where(stage_of < K - 1, o_idx + 1, -1)

    op_base_scaled = tables.job_base_tick.reshape(-1).astype(np.float32) / max(
        tables.p_ref_tick, 1.0
    )
    actual_machine = np.full(O, -1, dtype=np.int64)
    for j in range(J):
        n_sched = state.scheduled_op_count[j]
        for idx in range(n_sched):
            k = tables.job_route[j][idx]
            actual_machine[j * K + k] = int(state.operation_machine[(j, idx)])
    op_factory_index = np.array(
        [
            (-1 if state.assigned_factory[j] is None else int(state.assigned_factory[j]))
            for j in range(J)
            for _ in range(K)
        ],
        dtype=np.int64,
    )

    # ---- actions -----------------------------------------------------------
    a_job: list[int] = []
    a_fac: list[int] = []
    a_extra: list[tuple[float, float]] = []
    s_job: list[int] = []
    s_grid: list[int] = []
    s_mach: list[int] = []
    s_extra: list[tuple[float, float]] = []
    for action in actions:
        if is_assign(action):
            j, f = parse_assign(action)
            a_job.append(j)
            a_fac.append(f)
            a_extra.append(assignment_action_extra(state, sf, j, f))
        else:
            j, idx, m = parse_schedule(action)
            stage = tables.job_route[j][idx]
            s_job.append(j)
            s_grid.append(j * K + stage)
            s_mach.append(m)
            s_extra.append(scheduling_action_extra(state, j, idx, m))

    n_assign = len(a_job)
    n_schedule = len(s_job)

    def _col_int(vals: list[int], width: int) -> torch.Tensor:
        arr = np.full((1, max(width, 1)), 0, dtype=np.int64)
        if vals:
            arr[0, : len(vals)] = vals
        return torch.from_numpy(arr)

    def _col_extra(vals: list[tuple[float, float]], width: int) -> torch.Tensor:
        arr = np.zeros((1, max(width, 1), 2), dtype=np.float32)
        if vals:
            arr[0, : len(vals)] = np.asarray(vals, dtype=np.float32)
        return torch.from_numpy(arr)

    assign_job = _col_int(a_job, n_assign)
    assign_factory = _col_int(a_fac, n_assign)
    assign_extra = _col_extra(a_extra, n_assign)
    schedule_job = _col_int(s_job, n_schedule)
    schedule_op_grid = _col_int(s_grid, n_schedule)
    schedule_machine = _col_int(s_mach, n_schedule)
    schedule_extra = _col_extra(s_extra, n_schedule)

    # logit layout: [ assign block (width max(n_assign, 1)) | schedule block ]
    width_assign_block = int(assign_job.shape[1])
    action_mask = torch.zeros(
        1, width_assign_block + int(schedule_job.shape[1]), dtype=torch.bool
    )
    action_mask[0, :n_assign] = True
    action_mask[0, width_assign_block : width_assign_block + n_schedule] = True

    return GraphBatch(
        factory_features=factory_features,
        machine_features=machine_features,
        operation_features=operation_features,
        factory_mask=factory_mask,
        machine_mask=machine_mask,
        operation_mask=operation_mask,
        real_op_mask=real_op_mask,
        factory_machine_mask=fmm,
        operation_machine_mask=omm,
        predecessor_index=torch.from_numpy(pred.astype(np.int64)).unsqueeze(0),
        successor_index=torch.from_numpy(succ.astype(np.int64)).unsqueeze(0),
        op_job_index=torch.from_numpy(job_of.astype(np.int64)).unsqueeze(0),
        op_stage_index=torch.from_numpy(stage_of.astype(np.int64)).unsqueeze(0),
        op_factory_index=torch.from_numpy(op_factory_index).unsqueeze(0),
        op_base_scaled=torch.from_numpy(op_base_scaled.astype(np.float32)).unsqueeze(0),
        op_actual_machine=torch.from_numpy(actual_machine).unsqueeze(0),
        machine_speed=torch.from_numpy(tables.speed.astype(np.float32)).unsqueeze(0),
        machine_stage_index=torch.from_numpy(
            tables.machine_stage.astype(np.int64)
        ).unsqueeze(0),
        machine_factory_index=torch.from_numpy(
            tables.machine_factory.astype(np.int64)
        ).unsqueeze(0),
        assign_job=assign_job,
        assign_factory=assign_factory,
        assign_extra=assign_extra,
        schedule_job=schedule_job,
        schedule_op_grid=schedule_op_grid,
        schedule_machine=schedule_machine,
        schedule_extra=schedule_extra,
        action_mask=action_mask,
        n_assign=torch.tensor([n_assign], dtype=torch.int64),
        n_schedule=torch.tensor([n_schedule], dtype=torch.int64),
        actions=[list(actions)],
        n_jobs=(J,),
        n_stages=K,
    )


# ---------------------------------------------------------------------------
# batch collation
# ---------------------------------------------------------------------------
_TENSOR_PAD: dict[str, str] = {
    # field -> padding mode: "zero" | "minus1" | "maskfalse"
    "factory_features": "zero",
    "machine_features": "zero",
    "operation_features": "zero",
    "factory_mask": "maskfalse",
    "machine_mask": "maskfalse",
    "operation_mask": "maskfalse",
    "real_op_mask": "maskfalse",
    "factory_machine_mask": "maskfalse",
    "operation_machine_mask": "maskfalse",
    "predecessor_index": "minus1",
    "successor_index": "minus1",
    "op_job_index": "minus1",
    "op_stage_index": "minus1",
    "op_factory_index": "minus1",
    "op_base_scaled": "zero",
    "op_actual_machine": "minus1",
    "machine_speed": "zero",
    "machine_stage_index": "minus1",
    "machine_factory_index": "minus1",
    "assign_job": "minus1",
    "assign_factory": "minus1",
    "assign_extra": "zero",
    "schedule_job": "minus1",
    "schedule_op_grid": "minus1",
    "schedule_machine": "minus1",
    "schedule_extra": "zero",
    "action_mask": "maskfalse",
}


def collate_graphs(graphs: Sequence[GraphBatch]) -> GraphBatch:
    """Pad and concatenate per-state graphs into one batch (batch dim first)."""
    if not graphs:
        raise ValueError("collate_graphs requires at least one graph")
    K_set = {g.n_stages for g in graphs}
    if len(K_set) != 1:
        raise ValueError(f"cannot collate graphs with different num_stages: {K_set}")
    K = graphs[0].n_stages

    F = max(g.factory_features.shape[1] for g in graphs)
    M = max(g.machine_features.shape[1] for g in graphs)
    J = max(g.n_jobs[0] for g in graphs)
    O = J * K
    A1 = max(g.assign_job.shape[1] for g in graphs)     # assign block width
    A2 = max(g.schedule_job.shape[1] for g in graphs)   # schedule block width

    out: dict[str, Any] = {}
    for name, mode in _TENSOR_PAD.items():
        if name == "action_mask":
            continue  # re-laid-out below (block offsets differ per sample)
        sample = getattr(graphs[0], name)
        dtype = sample.dtype
        shape = list(sample.shape)
        shape[0] = len(graphs)

        if name in ("factory_features", "factory_mask", "factory_machine_mask"):
            shape[1] = F
            if name == "factory_machine_mask":
                shape[2] = M
        elif name in ("machine_features", "machine_mask", "machine_speed",
                      "machine_stage_index", "machine_factory_index"):
            shape[1] = M
        elif name in ("operation_features", "operation_mask", "real_op_mask",
                      "operation_machine_mask", "predecessor_index",
                      "successor_index", "op_job_index", "op_stage_index",
                      "op_factory_index", "op_base_scaled", "op_actual_machine"):
            shape[1] = O
            if name == "operation_machine_mask":
                shape[2] = M
        elif name in ("assign_job", "assign_factory", "assign_extra"):
            shape[1] = A1
        elif name in ("schedule_job", "schedule_op_grid", "schedule_machine",
                      "schedule_extra"):
            shape[1] = A2
        elif name == "action_mask":
            shape[1] = A1 + A2

        if mode == "maskfalse":
            filled = torch.zeros(shape, dtype=dtype)
        else:
            fill_value = -1 if mode == "minus1" else 0
            filled = torch.full(shape, fill_value, dtype=dtype)

        for b, g in enumerate(graphs):
            src = getattr(g, name)
            slices = [b] + [slice(0, src.shape[d]) for d in range(1, src.ndim)]
            filled[tuple(slices)] = src
        out[name] = filled

    action_mask = torch.zeros(len(graphs), A1 + A2, dtype=torch.bool)
    for b, g in enumerate(graphs):
        n1 = int(g.n_assign[0])
        n2 = int(g.n_schedule[0])
        action_mask[b, :n1] = True
        action_mask[b, A1 : A1 + n2] = True
    out["action_mask"] = action_mask

    out["n_assign"] = torch.tensor(
        [int(g.n_assign[0]) for g in graphs], dtype=torch.int64
    )
    out["n_schedule"] = torch.tensor(
        [int(g.n_schedule[0]) for g in graphs], dtype=torch.int64
    )
    out["actions"] = [list(g.actions[0]) for g in graphs]
    out["n_jobs"] = tuple(g.n_jobs[0] for g in graphs)
    out["n_stages"] = K
    return GraphBatch(**out)
