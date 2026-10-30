"""DHHFSP-HGNNPPO-Comparison.

Independent, makespan-only adaptation of
Bao, H., Pan, Q., Chew, C.-M., Wang, L., & Gao, L. (2026).
"Energy-Efficient Distributed Heterogeneous Hybrid Flow-Shop Scheduling Using
Graph Neural Network and Deep Reinforcement Learning". IEEE TASE 23, 3619-3635.

This package contains:
  * a data generator for DHHFSP instances with missing operations (``data``),
  * an event-driven simulation environment (``core``),
  * heterogeneous graph construction + HGNN encoders (``graph``),
  * the joint-action actor-critic and PPO training stack (``policy``, ``rl``),
  * a CP-SAT reference solver (``solvers``),
  * evaluation / statistics / plotting utilities (``evaluation``).
"""

__version__ = "0.1.0"
