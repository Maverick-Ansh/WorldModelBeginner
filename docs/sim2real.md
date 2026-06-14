# From Simulation to a Real Robotic Hand

The plan: **train in Isaac Lab (massively parallel GPU sim) → transfer to your hand.**
Training a world model from scratch on real hardware is unsafe and far too slow; sim gives
you millions of safe interactions, and Isaac Lab is built for sim-to-real.

> ⚠️ Safety first. A learned policy can command fast, high-torque motions. Always test with
> conservative torque/velocity limits, a hardware e-stop, and the hand mounted/clamped
> before anything is near it.

## 1. Install Isaac Lab

You need an NVIDIA RTX GPU. Follow the official guide (Isaac Sim + Isaac Lab):

- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

Verify a built-in task runs, e.g. the Shadow Hand cube-reorientation
`Isaac-Repose-Cube-Shadow-Direct-v0`, before bringing in this repo.

## 2. Train here on Isaac Lab

```bash
python -m worldmodel.train --algo dreamerv3 --env isaaclab_shadow_hand \
    env.num_envs=4096 prefill_steps=50000 total_steps=50000000
```

Notes:
- `worldmodel/envs/isaaclab.py` launches the Omniverse app, builds the registered task via
  `parse_env_cfg`, and maps its torch/GPU API onto our NumPy `Env` contract. Isaac Lab
  already auto-resets finished envs, matching our convention.
- **Throughput**: the first version round-trips tensors through NumPy for a single code
  path. For maximum speed you can keep everything on the GPU — replace the buffer's NumPy
  arrays with CUDA tensors and skip the conversions in the wrapper. This is the main
  optimization for large `num_envs`.
- **DreamerV3 vs TD-MPC2**: TD-MPC2's MPPI is expensive at thousands of envs (it plans for
  every env every step). Start with DreamerV3 at high `num_envs`, or run TD-MPC2 with fewer
  envs and more planning iterations.

## 3. Close the sim-to-real gap

Bake robustness in during training (configure in the Isaac Lab task cfg):

- **Domain randomization** — masses, friction, joint damping/stiffness, latency, control
  gains, sensor noise, object size/shape.
- **Observation/action parity** — make the sim observation vector contain exactly what the
  real hand can measure (joint positions/velocities, fingertip/contact sensors, object pose
  from your perception stack), and command actions the same way (e.g. position or torque
  targets) in the same order.
- **Action smoothing & limits** — penalize jerk/large actions in sim so the deployed policy
  is gentle; clamp to safe ranges.

## 4. Deploy to your hand

Implement one class — `worldmodel/envs/base.Env` — wrapping your hardware SDK:

```python
import numpy as np
from worldmodel.envs.base import Env, EnvSpec

class MyRealHand(Env):
    def __init__(self):
        self.num_envs = 1
        self.driver = connect_to_hand()          # your SDK
        self.spec = EnvSpec(
            obs_dim=...,                          # must match the trained policy
            act_dim=...,
            act_low=np.array(self.driver.joint_min, np.float32),
            act_high=np.array(self.driver.joint_max, np.float32),
            max_episode_steps=None,
        )

    def reset(self):
        self.driver.go_home()
        return self._read_obs()[None]            # shape [1, obs_dim]

    def step(self, action):                       # action in [-1, 1], shape [1, act_dim]
        cmd = self.spec.act_low + (np.clip(action[0], -1, 1) + 1) * 0.5 * (
            self.spec.act_high - self.spec.act_low)
        self.driver.send(cmd)                     # respect your control rate!
        obs = self._read_obs()[None]
        reward = np.array([self._reward()], np.float32)
        terminated = np.array([self._is_done()], bool)
        truncated = np.array([False], bool)
        return obs, reward, terminated, truncated, {}
```

Then run **inference** with the trained checkpoint (no learning on hardware to start):

```bash
python -m worldmodel.eval --algo dreamerv3 --env isaaclab_shadow_hand \
    --checkpoint logs/<run>/checkpoint.pt --episodes 5
```

…pointing the env config/factory at your `MyRealHand` (register it in
`worldmodel/envs/factory.py` under a new `backend`, e.g. `realhand`). Keep the **same obs
layout and action convention** used in sim, or the policy won't transfer.

## 5. (Optional) fine-tune on hardware
Once inference is safe and stable, you can cautiously fine-tune: collect a small amount of
real data into the replay buffer and run a *low* learning rate with tight safety limits.
Both agents are sample-efficient, which is exactly why a world-model approach suits real
robots.
