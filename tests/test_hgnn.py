"""HGNN / actor-critic tests (specification section 31)."""

from __future__ import annotations

import pytest
import torch

from dhhfsp_compare.core.instance import instance_from_dict
from dhhfsp_compare.core.simulator import Simulator
from dhhfsp_compare.data.generator_v2 import generate_instance
from dhhfsp_compare.graph.attention import GraphAttention
from dhhfsp_compare.graph.builder import build_graph, collate_graphs
from dhhfsp_compare.policy.bao_actor_critic import BaoActorCritic
from dhhfsp_compare.policy.action_selector import masked_log_softmax

MODEL_CFG = {
    "embedding_dim": 8,
    "hgnn_layers": 2,
    "actor_hidden_dim": 32,
    "critic_hidden_dim": 32,
}


def _model(seed: int = 0) -> BaoActorCritic:
    torch.manual_seed(seed)
    return BaoActorCritic(MODEL_CFG)


def _tiny_sim(problem_cfg, i: int, jlo: int = 3, jhi: int = 4) -> Simulator:
    d = generate_instance(problem_cfg, f"nn_{i}", seed=72000 + i, split="nn",
                          job_count_range=(jlo, jhi))
    return Simulator(instance_from_dict(d).validate())


def test_attention_empty_neighbour_rows_are_zero():
    torch.manual_seed(0)
    attn = GraphAttention(dim=4, edge_dim=2, use_self=False)
    query = torch.randn(1, 2, 4)
    key = torch.randn(1, 3, 4)
    edges = torch.randn(1, 2, 3, 2)
    mask = torch.zeros(1, 2, 3, dtype=torch.bool)
    mask[0, 1, 2] = True  # query 0 has no neighbours at all
    out = attn(query, key, mask, edge_features=edges)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0, 0], torch.zeros(4))
    assert out[0, 1].abs().sum() > 0


def test_attention_key_padding_does_not_change_output():
    torch.manual_seed(1)
    attn = GraphAttention(dim=4, edge_dim=None, use_self=True)
    query = torch.randn(1, 2, 4)
    key_real = torch.randn(1, 2, 4)
    mask = torch.ones(1, 2, 2, dtype=torch.bool)

    key_padded = torch.cat([key_real, torch.randn(1, 3, 4)], dim=1)
    mask_padded = torch.zeros(1, 2, 5, dtype=torch.bool)
    mask_padded[:, :, :2] = True
    out_a = attn(query, key_real, mask)
    out_b = attn(query, key_padded, mask_padded)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_forward_shapes_and_finiteness(manual_instance):
    model = _model()
    sim = Simulator(manual_instance)
    g = build_graph(sim.state, sim.legal_actions())
    logits, value = model.policy_value(g)
    assert logits.shape == (1, g.action_mask.shape[1])
    assert value.shape == (1,)
    assert torch.isfinite(logits[g.action_mask]).all()
    assert torch.isfinite(value).all()
    assert torch.isinf(logits[~g.action_mask]).all()


def test_padding_invariance_single_vs_batch(problem_cfg):
    model = _model()
    sim_big = _tiny_sim(problem_cfg, 1, 9, 10)
    sim_small = _tiny_sim(problem_cfg, 2, 3, 4)
    g_big = build_graph(sim_big.state, sim_big.legal_actions())
    g_small = build_graph(sim_small.state, sim_small.legal_actions())

    logits_b, value_b = model.policy_value(collate_graphs([g_big, g_small]))
    logits_big, value_big = model.policy_value(g_big)
    logits_small, value_small = model.policy_value(g_small)

    n1_big = int(g_big.n_assign)
    n2_big = int(g_big.n_schedule)
    n1_small = int(g_small.n_assign)
    n2_small = int(g_small.n_schedule)

    def take(logits, g, n1, n2):
        A1 = int(g.assign_job.shape[1])
        return torch.cat([logits[0, :n1], logits[0, A1:A1 + n2]])

    ref_big = take(logits_big, g_big, n1_big, n2_big)
    ref_small = take(logits_small, g_small, n1_small, n2_small)
    A1b = max(int(g_big.assign_job.shape[1]), int(g_small.assign_job.shape[1]))

    big_positions = list(range(n1_big)) + [A1b + p for p in range(n2_big)]
    small_positions = list(range(n1_small)) + [A1b + p for p in range(n2_small)]
    assert torch.allclose(logits_b[0, big_positions], ref_big, atol=1e-5)
    assert torch.allclose(logits_b[1, small_positions], ref_small, atol=1e-5)
    assert torch.allclose(value_b[0], value_big[0], atol=1e-6)
    assert torch.allclose(value_b[1], value_small[0], atol=1e-6)


def test_action_probabilities_are_padding_invariant(problem_cfg):
    model = _model()
    sim_big = _tiny_sim(problem_cfg, 3, 9, 10)
    sim_small = _tiny_sim(problem_cfg, 4, 3, 4)
    g_big = build_graph(sim_big.state, sim_big.legal_actions())
    g_small = build_graph(sim_small.state, sim_small.legal_actions())
    batched = collate_graphs([g_big, g_small])

    logits_b, _ = model.policy_value(batched)
    logits_small, _ = model.policy_value(g_small)
    probs_b = masked_log_softmax(logits_b, batched.action_mask)[1].exp()
    probs_small = masked_log_softmax(logits_small, g_small.action_mask)[0].exp()

    A1b = max(int(g_big.assign_job.shape[1]), int(g_small.assign_job.shape[1]))
    n1, n2 = int(g_small.n_assign), int(g_small.n_schedule)
    batched_probs = torch.cat([probs_b[:n1], probs_b[A1b:A1b + n2]])
    single_probs = torch.cat(
        [probs_small[:n1], probs_small[g_small.assign_job.shape[1]:g_small.assign_job.shape[1] + n2]]
    )
    assert torch.allclose(batched_probs, single_probs, atol=1e-5)
    assert abs(float(batched_probs.detach().sum()) - 1.0) < 1e-4


def test_only_one_action_kind_and_both_kinds(manual_instance):
    model = _model()
    sim = Simulator(manual_instance)
    # initial state: assign actions only (no job is assigned yet)
    g = build_graph(sim.state, sim.legal_actions())
    assert int(g.n_schedule) == 0 and int(g.n_assign) == 4
    logits, value = model.policy_value(g)
    assert torch.isfinite(logits[g.action_mask]).all()

    # assign both jobs -> schedule-only state
    sim.step(("assign", 0, 0))
    sim.step(("assign", 1, 1))
    g = build_graph(sim.state, sim.legal_actions())
    assert int(g.n_assign) == 0 and int(g.n_schedule) == 2
    logits, value = model.policy_value(g)
    assert torch.isfinite(logits[g.action_mask]).all()

    # mixed state: assign job 1, then job 0 has scheduling actions too
    sim2 = Simulator(manual_instance)
    sim2.step(("assign", 0, 0))
    g = build_graph(sim2.state, sim2.legal_actions())
    assert int(g.n_assign) == 2 and int(g.n_schedule) == 1
    logits, value = model.policy_value(g)
    assert torch.isfinite(logits[g.action_mask]).all()


def test_gradients_reach_all_components(manual_instance):
    model = _model()
    sim = Simulator(manual_instance)
    sim.step(("assign", 0, 0))  # mixed state: assign + schedule actions
    g = build_graph(sim.state, sim.legal_actions())
    assert int(g.n_assign) > 0 and int(g.n_schedule) > 0
    logits, value = model.policy_value(g)
    loss = logits[g.action_mask].pow(2).mean() + value.pow(2).mean()
    loss.backward()

    def has_grad(prefix: str) -> bool:
        return any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in model.named_parameters()
            if n.startswith(prefix)
        )

    assert has_grad("encoder.")
    assert has_grad("assign_scorer.")
    assert has_grad("schedule_scorer.")
    assert has_grad("critic.")
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert no_grad == [], f"parameters without gradients: {no_grad}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_consistency(device, manual_instance):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = _model().to(device)
    sim = Simulator(manual_instance)
    g = build_graph(sim.state, sim.legal_actions()).to(device)
    logits, value = model.policy_value(g)
    assert logits.device.type == device and value.device.type == device
    logits_cpu, value_cpu = _model().policy_value(
        build_graph(sim.state, sim.legal_actions())
    )
    assert torch.allclose(logits.cpu(), logits_cpu, atol=1e-5)
    assert torch.allclose(value.cpu(), value_cpu, atol=1e-6)
