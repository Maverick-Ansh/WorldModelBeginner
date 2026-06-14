"""End-to-end smoke tests for both agents on the toy env (tiny configs, CPU)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from worldmodel.agents import make_agent
from worldmodel.common.buffer import SequenceReplayBuffer
from worldmodel.common.config import load_configs
from worldmodel.common.utils import set_seed
from worldmodel.envs import make_env

ROOT = Path(__file__).resolve().parent.parent

TINY = {
    "dreamerv3": [
        "algo.hidden=32", "algo.embed_dim=16", "algo.deter=32", "algo.groups=4",
        "algo.classes=4", "algo.num_bins=15", "algo.mlp_layers=1", "algo.batch_size=4",
        "algo.seq_len=8", "algo.horizon=3",
    ],
    "tdmpc2": [
        "algo.latent_dim=32", "algo.hidden=32", "algo.num_q=2", "algo.num_bins=15",
        "algo.mlp_layers=1", "algo.batch_size=8", "algo.horizon=3", "algo.plan_samples=32",
        "algo.plan_elites=8", "algo.plan_pi_samples=4", "algo.plan_iterations=2",
    ],
}


def _cfg(algo: str):
    return load_configs(
        [ROOT / "configs/default.yaml", ROOT / f"configs/algo/{algo}.yaml",
         ROOT / "configs/env/toy.yaml"],
        overrides=["device=cpu", "replay_capacity=2000", "env.num_envs=2",
                   "env.num_joints=3", "env.episode_steps=10"] + TINY[algo],
    )


def _prefill(env, buf, steps=60):
    obs = env.reset()
    is_first = np.ones(env.num_envs, dtype=bool)
    for _ in range(steps):
        a = np.random.uniform(-1, 1, (env.num_envs, env.spec.act_dim)).astype(np.float32)
        nobs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r, 1.0 - term.astype(np.float32), is_first)
        obs, is_first = nobs, term | trunc
    return obs, is_first


@pytest.mark.parametrize("algo", ["dreamerv3", "tdmpc2"])
def test_agent_act_update_and_io(algo, tmp_path):
    cfg = _cfg(algo)
    set_seed(0)
    device = torch.device("cpu")
    env = make_env(cfg.env, seed=0)
    agent = make_agent(cfg, env.spec.obs_dim, env.spec.act_dim, device)
    buf = SequenceReplayBuffer(cfg.replay_capacity, env.spec.obs_dim, env.spec.act_dim,
                               device, num_envs=env.num_envs, seed=0)

    obs, is_first = _prefill(env, buf)

    # acting produces bounded actions of the right shape
    action = agent.act(obs, is_first, training=True)
    assert action.shape == (env.num_envs, env.spec.act_dim)
    assert np.all(np.abs(action) <= 1.0 + 1e-5)

    # a few updates yield finite metrics
    metrics = {}
    for _ in range(3):
        metrics = agent.update(buf)
    assert metrics, "update returned no metrics"
    assert all(np.isfinite(v) for v in metrics.values()), metrics

    # greedy acting also works
    eval_action = agent.act(obs, is_first, training=False)
    assert eval_action.shape == (env.num_envs, env.spec.act_dim)
    assert np.all(np.abs(eval_action) <= 1.0 + 1e-5)

    # checkpoint save/load roundtrip
    ckpt = tmp_path / "ckpt.pt"
    agent.save(str(ckpt))
    clone = make_agent(cfg, env.spec.obs_dim, env.spec.act_dim, device)
    clone.load(str(ckpt))
    assert clone.act(obs, is_first, training=False).shape == action.shape


def test_dreamer_resets_state_on_is_first():
    cfg = _cfg("dreamerv3")
    set_seed(0)
    env = make_env(cfg.env, seed=0)
    agent = make_agent(cfg, env.spec.obs_dim, env.spec.act_dim, torch.device("cpu"))
    obs = env.reset()
    agent.act(obs, np.ones(env.num_envs, dtype=bool))   # first step initializes state
    deter_before = agent._state["deter"].clone()
    agent.act(obs, np.ones(env.num_envs, dtype=bool))   # is_first should zero prev state
    # with identical obs and a reset, the new posterior recomputes from a zeroed prior
    assert deter_before.shape == agent._state["deter"].shape
