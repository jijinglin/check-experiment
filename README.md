# DHHFSP-HGNNPPO-Comparison

An independent, makespan-only implementation of

> Bao, H., Pan, Q., Chew, C.-M., Wang, L., & Gao, L. (2026). *Energy-Efficient
> Distributed Heterogeneous Hybrid Flow-Shop Scheduling Using Graph Neural
> Network and Deep Reinforcement Learning*. IEEE TASE, 23, 3619–3635.
> DOI: 10.1109/TASE.2025.3591984

adapted to the DHHFSP scheduling problem **with missing (skipped) operations**,
built to compare its overall scheduling quality, cross-scale generalization and
computational efficiency against an externally trained **Concede-HRL** model on
one shared static problem definition.

This project was implemented from a written engineering specification
(not from the original DHHFSP codebase); it neither imports nor modifies any
other project, and every artifact (datasets, checkpoints, results) lives inside
this directory.  All numbers in any report are produced by the commands below —
nothing is hard-coded or fabricated.

---

## 1. Install

Python ≥ 3.11.  Development environment: CPython 3.12.7 on Windows, CPU-only
torch.  From the project root:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```

## 2. Quick start (the full pipeline)

```bash
# 1) freeze all dataset splits (train / validation / test_* / ood_*)
python scripts/generate_dataset.py --config configs/problem_v2.yaml

# 2) unit + acceptance tests
pytest -q                 # fast suite
pytest -q -m slow         # 300 random instances, random policy, no deadlock

# 3) small smoke training (a few seconds, uses data/smoke)
python scripts/train_bao.py --config configs/smoke.yaml

# 4) main experiment: five independent training seeds (9600 episodes each)
python scripts/train_bao.py --config configs/bao_hgnn_ppo.yaml --seed 42
#    ... repeat for 123, 456, 789, 2026   (each ~seeds' run_dir: runs/bao/seed_<seed>)
#    resume after an interruption:
python scripts/train_bao.py --config configs/bao_hgnn_ppo.yaml --seed 42 \
    --resume runs/bao/seed_42/last.pt

# 5) evaluate a trained checkpoint on the frozen test splits
python scripts/evaluate_bao.py --checkpoint runs/bao/seed_42/best.pt \
    --split test_small,test_medium,test_large,ood_50,ood_60,ood_80

# 6) import the externally produced Concede-HRL reference results
python scripts/import_concede.py --input reference/concede/predictions.csv

# 7) comparison statistics, tables and figures
python scripts/run_comparison.py --config configs/experiments.yaml

# 8) paper tables + report
python scripts/build_report.py --output results/reports/
```

Optional solution-quality reference (OR-Tools CP-SAT, specification §26):

```bash
python -m dhhfsp_compare.solvers.cp_sat --split test_small --time-limit 60 \
    --out results/cp_sat_test_small.csv
```

## 3. Repository layout

```
configs/      problem definition, training config, experiment plan, smoke config
src/dhhfsp_compare/
  core/       types, instance, validator, event-driven simulator, reward estimator
  data/       generator_v2 (v2-1), schema/hashing, dataset IO, manifests
  graph/      state features, graph builder (+batch collation), attention, HGNN
  policy/     BaoActorCritic (joint actor + single critic), action selector
  rl/         rollout collection, buffer, PPO update, trainer
  solvers/    CP-SAT reference solver
  evaluation/ checkpoint evaluation, Concede import, statistics, plots
  utils/      seeding, checkpoints, logging
scripts/      the eight CLI entry points listed above
tests/        unit + acceptance tests (pytest)
data/         frozen instances + per-split manifests (generated)
reference/    Concede-HRL reference results go here (see its README)
runs/         training runs (checkpoints, logs, validation curves)
results/      raw per-instance results, tables, figures, reports
```

## 4. Problem definition and dataset

Static DHHFSP: 3 factories × 5 global stages × 45 machines (per-factory machine
counts `[[2,2,5,4,2],[3,3,3,3,3],[5,4,1,2,3]]`), heterogeneous machine speeds
drawn once from `{1.1,1.4,1.7,2.0}` with seed `20260707`, product variants whose
routes may miss 0–2 non-first stages, bottleneck/secondary/light stage time
ranges, and 16–80 jobs depending on the split.  The generator (`v2-1`) consumes
its RNG in one exact documented order, so the same seed always reproduces the
same instance byte-for-byte.

Time is integer **ticks** (0.1 time units); `duration_tick = round(base_time ×
round(speed × 10))` in ticks, which is exact for these one-decimal speeds.
Makespans are reported in problem time units (ticks / 10).

Every instance is frozen with a SHA-256 content hash; each split has a manifest
(`data/manifests/<split>.json`) with instance ids + hashes, problem-config hash
and generator version.  Training checkpoints, evaluation results and imported
reference rows are all bound to these hashes; a mismatch aborts.

| split       | jobs  | instances | seeds       |
|-------------|-------|-----------|-------------|
| train       | 16–40 | 200       | 42–241      |
| validation  | 16–40 | 20        | 1242–1261   |
| test_small  | 16–20 | 100       | 10000–10099 |
| test_medium | 21–30 | 100       | 11000–11099 |
| test_large  | 31–40 | 100       | 12000–12099 |
| ood_50      | 50    | 100       | 13000–13099 |
| ood_60      | 60    | 100       | 14000–14099 |
| ood_80      | 80    | 100       | 15000–15099 |

Datasets are frozen **before** any training; test splits are never used for
checkpoint selection and are never modified based on any method's performance.

## 5. Method summary (HGNN-PPO adaptation)

* **Mixed joint action space** — at every decision point the policy scores
  `("assign", job, factory)` and `("schedule", op, machine)` candidates in ONE
  softmax over the concatenated logits.  There is no high-level/low-level split
  and no WAIT action; when no action is legal the simulator advances physical
  time through the completion-event heap (empty heap + unfinished schedule
  raises `DeadlockError`).
* **Heterogeneous graph** — Factory / Machine / Operation nodes on a fixed
  `[jobs × stages]` grid (virtual nodes for missing stages, flagged by
  `real_op_mask`), Factory–Machine, Operation–Machine (with processing-time +
  actually-selected edge features) and grid-adjacent Operation–Operation
  relations; padded batch dimension with strict padding invariance.
* **HGNN** — two rounds of factory → machine → operation updates; the
  operation update fuses five channels (self / predecessor / successor /
  machine-attention / assigned-factory).  Pure PyTorch masked attention — no
  compiled graph-extension dependency.
* **Actor–critic** — two per-kind scoring MLPs feeding one joint masked
  distribution plus a **single** critic over the global graph embedding
  (`[mean factory ‖ mean machine ‖ mean operation]`, 3·8 dims).
* **Reward** — telescoping makespan-estimate improvement
  `r_t = (Ĉ(s_t) − Ĉ(s_{t+1})) / H_ref`; total return over an episode equals
  `(Ĉ(s_0) − Cmax) / H_ref` exactly (tested to 1e-6).  No extra terminal
  penalty.
* **PPO** — clipped objective on the joint distribution's log-probs, one ratio
  per transition (never per action kind), Return-to-Go per episode, γ = 1,
  advantage normalization over the rollout batch.

## 6. Training protocol

| item | value |
|---|---|
| episodes per update | 20 |
| total episodes | 9600 (480 updates) per seed |
| training seeds | 42, 123, 456, 789, 2026 |
| learning rate | 2e-4 (Adam) |
| epochs / minibatch | 3 / 256 transitions (minibatch size is an adaptation) |
| validation | every 10 updates, deterministic (argmax) on the frozen validation split |
| checkpoint selection | `best_validation` (test sets never influence checkpoints) |

Concede-HRL's matched budget is 9600 episodes (8 episodes × 1200 updates
internally); each algorithm keeps its own internal PPO schedule.  Sample
counts, decision counts and wall-clock times are logged separately.

Training uses the fixed train split with a no-replacement shuffle cycle driven
by a per-seed RNG.  Checkpoints (`best.pt`, `last.pt`) store model + optimizer
state, update/episode counters, best validation value, the full config, all RNG
states (python/numpy/torch + the sampling generator), the sampler state and the
dataset fingerprint; `--resume` continues exactly where it stopped (verified in
`tests/test_ppo.py::test_resume_reproduces_uninterrupted_run`).

Run artifacts per seed (`runs/bao/seed_<seed>/`):
`train.log`, `episodes.jsonl` (per-episode makespan/reward/decisions),
`validation.csv` (per-instance validation results), `best.pt`, `last.pt`.

`runs/bao_smoke/` and `runs/readiness_check/` are small engineering-check runs
(smoke config; and a 1-update run of the real config) — they are **not**
experiment results.

## 7. Evaluation and the Concede-HRL comparison

* `evaluate_bao.py` runs deterministic episodes per test instance and records
  `instance_id, instance_hash, algorithm, training_seed, checkpoint_id,
  makespan, inference_time_ms, num_policy_decisions, num_assign_actions,
  num_schedule_actions, schedule_valid`.  A makespan is only recorded after
  the schedule passes the project's own `validate_schedule`; failed schedules
  are hard errors, never silently dropped.
* Concede-HRL results are **imported** (`import_concede.py`, see
  `reference/concede/README.md`), never re-implemented here.  The importer
  enforces: `method == Concede-HRL`, `objective == makespan`, a concrete
  `evaluation_mode`, per-instance SHA-256 equality with the frozen dataset,
  full per-instance coverage for every declared seed, and — when schedule
  files are supplied — re-validation with recomputed makespans.
* `run_comparison.py` computes per-scale summaries, paired per-instance
  comparisons, per-seed statistics, Wilcoxon signed-rank tests and stratified
  bootstrap CIs.  Improvement sign convention (specification §24, implemented
  literally): `Imp = (Bao − Concede) / Bao × 100 %` — **negative ⇒ Concede-HRL
  lower makespan, positive ⇒ HGNN-PPO lower makespan**.
* A single Concede training seed is allowed for import but flagged in
  warnings, in `comparison_summary.json` and in the generated report's
  Limitations section; it is never presented as equal-replication evidence
  against the five HGNN-PPO seeds.

## 8. Tests

```bash
pytest -q            # unit suite: schema/hash round-trip, validator, generator
                     # determinism, the hand-verified 32.4 example, mixed action
                     # space, missing operations, graph/attention/HGNN,
                     # padding invariance, reward telescoping, PPO, checkpoint
                     # resume exactness, reference-import strictness, e2e pipeline
pytest -q -m slow    # Phase-1 acceptance: 300 random instances, random legal
                     # policy, zero deadlock, all schedules valid
```

The hand-verifiable example from the specification (2 factories × 3 stages,
Cmax = 32.4 under the prescribed fixed assignment) is asserted interval-exact
in `tests/test_simulator.py`; the CP-SAT optimum of the same instance (28.8,
free assignment) is asserted too.

## 9. Adaptation notes (what is, and is not, from the paper)

This project implements the *makespan-only* adaptation requested by its
specification.  **Not implemented** (deliberately, per specification §0.2/§41):
energy objectives, machine on/off dynamics, multi-objective evaluation,
hierarchical high-level/low-level decision structures, Concede-style WAIT
mechanisms, per-kind action normalizations.

The following are engineering choices of this project where the paper's exact
detail was not available; they are marked here so that nothing in this
repository is misattributed to the original paper:

* ELU activations, 2-layer MLPs (~128 hidden, per config) for scorers/critic,
  Adam with lr 2e-4, PPO minibatch size 256, entropy coefficient 0.01 —
  pre-registered starting parameters, not claimed as paper originals.
* `H_ref = max_k Σ_j p_base / Σ_{m∈k} 1/q_m` and the exact reward/estimator
  form follow the project specification (§18); estimate features are
  normalized by `H_ref`, `P_ref`, `q_max` as specified (§10).
* Operation precedence in the graph follows grid adjacency through virtual
  (missing-stage) nodes; physical execution connects adjacent **real**
  operations only (§8.2).  The dense op–op mask of the specification is stored
  as predecessor/successor index vectors for memory reasons (exact — at most
  one neighbour per direction).
* Operation–machine edge features (processing time / P_ref, actually-selected
  flag) are stored factorized and materialized lazily; the attention's edge
  term is computed exactly, without materializing `[U, V, d]` intermediates.
* Two extra candidate features per action kind exactly as specified (§15):
  factory path / factory load (assignment), processing time / machine
  availability gap (scheduling).
* Sampling uses `torch.multinomial` on the policy's device; CPU runs are
  bitwise reproducible, CUDA sampling may vary across devices/driver versions.
* Deterministic evaluation = argmax over the joint masked distribution.

Any paper text produced from this repository should re-check the message-passing
formulas of §13 against the published paper and mark the adaptations above
explicitly.

## 10. Provenance & traceability

* Every result row carries instance id + instance hash + training seed +
  checkpoint id; `results/tables/comparison_summary.json` additionally records
  the dataset fingerprint, the methods' seeds and the reference-import
  warnings.
* `build_report.py` regenerates all paper tables and the markdown report from
  the raw per-instance results — no numbers are entered by hand.
* Configs, generator version (`v2-1`), network shape and the dataset fingerprint
  must be quoted in the paper's reproducibility statement; changing any of
  them invalidates checkpoints against older data on purpose (`DatasetError` /
  dataset-hash refusal on resume/import).
