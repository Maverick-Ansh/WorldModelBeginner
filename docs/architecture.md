# Architecture

This document explains how the pieces fit together and the key design decisions. The code
is intentionally compact and commented; read it alongside this doc.

## Shared contracts

### Environment (`worldmodel/envs/base.py`)
A single `Env` interface makes one training loop work from a toy task up to Isaac Lab:

- **Vectorized** — every array has a leading `num_envs` dimension (1 for toy/Gym).
- **Auto-resetting** — `step` resets finished envs internally and returns the next
  episode's first observation, so the driver never special-cases resets.
- **Normalized actions** — agents always emit actions in `[-1, 1]^act_dim`; each backend
  rescales to the true actuator range. Only this rescaling changes between sim and a real
  hand, which is what makes a learned policy portable.

### Replay (`worldmodel/common/buffer.py`)
World models train on short *trajectories*. `SequenceReplayBuffer` stores
`[capacity, num_envs, ...]` and samples contiguous sub-sequences along time, keeping each
env's stream clean. `is_first` flags mark episode boundaries inside a window so recurrent
state can be reset where needed.

### Numerics (`worldmodel/common/distributions.py`)
The tricks that let both agents use fixed hyper-parameters across reward scales:

- **symlog / symexp** — compress wide-range targets: `symlog(x) = sign(x)·log(1+|x|)`.
- **Two-hot discrete regression** (`SymlogTwoHot`) — predict reward/value as a softmax over
  bins uniformly spaced in symlog space; train with cross-entropy against the two-hot
  encoding of the target. Far more stable than MSE.
- **TanhNormal** — squashed Gaussian policy over `[-1, 1]`.
- **SimNorm** (`common/nets.py`) — projects TD-MPC2's latent onto a product of simplices,
  keeping it bounded and sparse.

---

## DreamerV3 (`worldmodel/agents/dreamer`)

Learns a latent world model, then learns behavior **entirely inside imagined rollouts**.

### World model — Recurrent State-Space Model (`rssm.py`, `world_model.py`)
Each latent state has a deterministic recurrent part `h` (LayerNorm GRU) and a stochastic
part `z` from `groups × classes` categoricals (straight-through gradients, 1% uniform mix).

- **Prior** `p(z_t | h_t)` — the model's imagination of the next latent.
- **Posterior** `q(z_t | h_t, x_t)` — corrected by the actual observation.

Trained to jointly:
1. reconstruct the observation (`SymlogMSE` decoder),
2. predict reward (two-hot) and episode continuation (Bernoulli), and
3. minimize **KL(posterior ‖ prior)** with *free bits* (1 nat) and *KL balancing*
   (`dyn` 0.5 / `rep` 0.1) so the prior chases the posterior without collapsing.

### Behavior — actor-critic in imagination (`behavior.py`)
From every observed latent, the RSSM dreams `horizon` steps forward using the actor.
The critic regresses **λ-returns** (two-hot), with an EMA *slow critic* providing stable
bootstrap targets. The actor maximizes normalized λ-returns plus an entropy bonus, with
gradients flowing back through the differentiable dynamics (analytic policy gradient).
Returns are normalized by an EMA of their 5–95th percentile spread (`Moments`) so the
actor sees a unit scale regardless of task reward magnitude.

```
encode obs ─▶ RSSM.observe ─▶ latent states ─▶ imagine(actor, H) ─▶ λ-returns
                   │                                   │
              world-model loss                   actor & critic loss
```

---

## TD-MPC2 (`worldmodel/agents/tdmpc2`)

A **decoder-free** latent model learned for control, with planning at decision time.

### Model (`world_model.py`)
- `encoder`: obs → latent `z` (MLP + SimNorm).
- `dynamics`: `(z, a) → z'`.
- `reward`: `(z, a) →` reward (two-hot).
- `Q-ensemble` (5): `(z, a) →` value (two-hot); targets use the **min over 2 random**
  members (clipped double-Q).
- `policy prior π`: amortized actor used to warm-start planning and bound its quality.

There is no observation decoder; the latent is shaped by a **self-consistency** loss —
the predicted next latent must match the (target-encoder) encoding of the next obs —
plus the reward and TD-value losses, combined with per-step horizon weighting `ρ^t`. A
target network (EMA of encoder/dynamics/Q) stabilizes the TD targets.

### Planning — MPPI (`planner.py`)
At each step the planner imagines many action sequences through the latent dynamics,
scores them by discounted predicted reward + a terminal Q-value, and refits a per-step
Gaussian toward the high-return *elites* over several iterations. A fraction of candidates
come from the policy prior. Fully vectorized over `num_envs`; warm-started from the
previous step's plan.

```
encode obs ─▶ z0 ─▶ MPPI{ sample → roll dynamics → score → refit }×iters ─▶ action
                         ▲ policy-prior seeds
```

---

## Training loop (`train.py`)
Collect with the current policy (random during prefill) → store transitions → sample and
update. `cont = 1 − terminated` (time-limit truncation does **not** zero the discount, so
value bootstrapping at the horizon is correct). Metrics stream to TensorBoard and console;
checkpoints are written periodically.

## Extending
- **New algorithm**: implement `agents/base.Agent`, register it in `agents/factory.py`,
  add a `configs/algo/<name>.yaml`.
- **New environment / real hand**: implement `envs/base.Env`, register in
  `envs/factory.py`, add `configs/env/<name>.yaml`. See `docs/sim2real.md`.
- **Pixel observations**: add a CNN encoder/decoder in `common/nets.py` and swap the
  vector encoder in each world model — the rest of the pipeline is unchanged.
