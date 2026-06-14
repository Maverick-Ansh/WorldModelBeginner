"""WorldModelBeginner: modular world-model RL for dexterous robotic-hand control.

Two interchangeable model-based agents share a single env/replay/training harness:

* :mod:`worldmodel.agents.dreamer`  -- DreamerV3 (RSSM + actor-critic in imagination)
* :mod:`worldmodel.agents.tdmpc2`   -- TD-MPC2 (latent dynamics + MPPI planning)

Environments are hidden behind :class:`worldmodel.envs.base.Env`, so the same code
runs on a dependency-free toy env, Gymnasium/MuJoCo, NVIDIA Isaac Lab, or your real
robotic hand -- you only implement one interface. See ``docs/sim2real.md``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
