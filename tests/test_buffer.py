import numpy as np
import torch

from worldmodel.common.buffer import SequenceReplayBuffer


def test_sample_shapes_and_env_separation():
    num_envs, obs_dim, act_dim = 3, 2, 1
    buf = SequenceReplayBuffer(capacity=300, obs_dim=obs_dim, act_dim=act_dim,
                               device=torch.device("cpu"), num_envs=num_envs, seed=0)
    # obs encodes (timestep, env_id) so we can verify contiguity and separation.
    for t in range(50):
        obs = np.stack([[t, e] for e in range(num_envs)]).astype(np.float32)
        action = np.full((num_envs, act_dim), t, dtype=np.float32)
        buf.add(obs, action, reward=np.zeros(num_envs), cont=np.ones(num_envs),
                is_first=np.zeros(num_envs))

    batch = buf.sample(batch_size=16, seq_len=6)
    assert batch["obs"].shape == (16, 6, obs_dim)
    assert batch["action"].shape == (16, 6, act_dim)
    assert batch["reward"].shape == (16, 6)

    obs = batch["obs"].numpy()
    # within each sampled sequence: timesteps are consecutive ...
    assert np.all(np.diff(obs[:, :, 0], axis=1) == 1)
    # ... and the env id is constant (sequences never mix envs).
    assert np.all(np.diff(obs[:, :, 1], axis=1) == 0)


def test_ring_overwrite_keeps_contiguity():
    buf = SequenceReplayBuffer(capacity=20, obs_dim=1, act_dim=1,
                               device=torch.device("cpu"), num_envs=1, seed=0)
    for t in range(100):  # force several wraps
        buf.add(np.array([[t]], np.float32), np.array([[0]], np.float32),
                np.zeros(1), np.ones(1), np.zeros(1))
    batch = buf.sample(batch_size=8, seq_len=5)
    obs = batch["obs"].numpy()
    assert np.all(np.diff(obs[:, :, 0], axis=1) == 1)  # no boundary crossing
    assert len(buf) == 20
