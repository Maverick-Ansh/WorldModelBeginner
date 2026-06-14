# WorldModelBeginner 🦾

Modular, **world-class world-model reinforcement learning** for dexterous robotic-hand
control — built to train in simulation and transfer to a real hand (the Iron-Man-style
goal). Two state-of-the-art model-based agents share one clean training harness:

| Agent | Idea | Best for |
|------|------|----------|
| **DreamerV3** | Learns a latent world model (RSSM) and trains an actor-critic entirely *in imagination*. | General, robust, sample-efficient learning across domains. |
| **TD-MPC2** | Learns a latent dynamics + value model and **plans** with MPPI at decision time. | Top raw performance on continuous control & dexterous manipulation. |

Everything is vectorized for **NVIDIA Isaac Lab** (thousands of GPU-parallel hands), with
a dependency-free toy env so you can run the whole stack on a laptop CPU in seconds.

> Status: reference implementations, validated end-to-end on the toy env (see
> [Testing](#testing)). Designed to be *tweaked* — clear code, fixed-hyperparameter
> recipes, one interface to swap in your own robot.

---

## Why world models?

A *world model* learns to predict how the environment evolves: given the current state
and an action, what happens next, and what reward follows. Once you have that, the agent
can learn behavior by "dreaming" (DreamerV3) or by "planning" through the model (TD-MPC2)
instead of needing millions of real interactions. For a real robotic hand — where data is
expensive and mistakes break hardware — this sample efficiency is exactly what you want.

---

## Install

```bash
git clone <this-repo> && cd WorldModelBeginner
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core (PyTorch CPU/GPU, gymnasium, etc.)
# optional extras:
pip install -e ".[mujoco]"  # Gymnasium-Robotics dexterous hands (Adroit / Shadow)
pip install -e ".[dev]"     # pytest + ruff
```

For a CUDA build of PyTorch, follow https://pytorch.org/get-started/locally/.
For Isaac Lab, see [`docs/sim2real.md`](docs/sim2real.md).

## Quickstart (CPU, ~seconds)

```bash
# DreamerV3 on the dependency-free toy hand
python -m worldmodel.train --algo dreamerv3 --env toy

# TD-MPC2, with a couple of tweaks via key=value overrides
python -m worldmodel.train --algo tdmpc2 --env toy seed=1 algo.horizon=3

# watch it learn
tensorboard --logdir logs

# evaluate a checkpoint with the greedy policy
python -m worldmodel.eval --algo tdmpc2 --env toy \
  --checkpoint logs/<run>/checkpoint.pt --episodes 10
```

Train on a real dexterous-hand benchmark (needs `.[mujoco]`):

```bash
python -m worldmodel.train --algo dreamerv3 --env adroit env.task=AdroitHandDoor-v1
```

Train on Isaac Lab (needs a GPU + Isaac Sim/Lab — see the doc):

```bash
python -m worldmodel.train --algo dreamerv3 --env isaaclab_shadow_hand \
  prefill_steps=50000 total_steps=50000000
```

---

## How it fits together

```
            ┌──────────────┐   actions in [-1,1]   ┌──────────────────────────┐
            │   Env (any)  │◀──────────────────────│         Agent            │
 toy / gym /│  vectorized, │   obs, reward, done    │  DreamerV3  or  TD-MPC2  │
 isaaclab / │ auto-reset   │──────────────────────▶│                          │
 real hand  └──────┬───────┘                        └────────────┬─────────────┘
                   │ transitions                                 │ samples
                   ▼                                             ▼
            ┌─────────────────────────── SequenceReplayBuffer ───────────────────┐
            │   [capacity, num_envs, ...]  ·  samples contiguous sub-sequences    │
            └─────────────────────────────────────────────────────────────────────┘
```

* **One env interface** ([`worldmodel/envs/base.py`](worldmodel/envs/base.py)) — vectorized,
  auto-resetting, normalized actions. Backends: `toy`, `gym`, `isaaclab`. Your real hand =
  one more implementation of this interface.
* **One agent interface** ([`worldmodel/agents/base.py`](worldmodel/agents/base.py)) —
  `act()` / `update()` / save / load.
* **Shared components** ([`worldmodel/common`](worldmodel/common)) — config, networks
  (MLP / SimNorm / LayerNorm-GRU), distributions (symlog, two-hot, tanh-Normal), the
  replay buffer, and logging.

See [`docs/architecture.md`](docs/architecture.md) for the math and design rationale, and
[`docs/sim2real.md`](docs/sim2real.md) for the Isaac Lab → real-hardware path.

## Configuration

Plain YAML, deep-merged in this order then overridden by `key.subkey=value` CLI args:

```
configs/default.yaml          # run-level: steps, seeds, replay, logging
configs/algo/{dreamerv3,tdmpc2}.yaml
configs/env/{toy,adroit,isaaclab_shadow_hand}.yaml
```

```bash
# tweak anything from the command line
python -m worldmodel.train --algo tdmpc2 --env toy \
    algo.horizon=5 algo.plan_samples=1024 env.num_envs=8 total_steps=200000
```

## Which algorithm should I use?

* **Start with DreamerV3** if you want a single robust recipe that "just works" across
  tasks, learns from a single GPU, and you value sample efficiency from imagination.
* **Reach for TD-MPC2** when you want the strongest possible control performance and can
  afford planning compute at deploy time; its MPPI planner often shines on contact-rich,
  dexterous tasks. (Planning is costly at thousands of parallel envs — see the config notes.)

## Project layout

```
worldmodel/
  common/      config · nets · distributions · buffer · logger · utils
  envs/        base · toy · gym_wrapper · isaaclab · factory
  agents/
    dreamer/   rssm · world_model · behavior · agent
    tdmpc2/    world_model · planner · agent
  train.py · eval.py
configs/   ·   tests/   ·   docs/   ·   scripts/
```

## Testing

```bash
pip install -e ".[dev]"
pytest            # fast CPU smoke + unit tests for both agents
```

The suite trains tiny versions of both agents end-to-end on the toy env, and checks the
replay buffer, distributions, env auto-reset, config merging, and checkpoint I/O.

## References

- Hafner, Pasukonis, Ba, Lillicrap. *Mastering Diverse Domains through World Models* (DreamerV3), 2023.
- Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control*, 2024.
- NVIDIA *Isaac Lab* — https://isaac-sim.github.io/IsaacLab/

This is an independent educational reimplementation, not affiliated with the original authors.

## License

MIT — see [LICENSE](LICENSE).
