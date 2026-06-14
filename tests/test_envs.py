import numpy as np

from worldmodel.envs import make_env
from worldmodel.envs.toy import ToyHandEnv


def test_toy_env_shapes_and_spec():
    env = ToyHandEnv(num_envs=4, num_joints=5, episode_steps=10)
    assert env.spec.obs_dim == 15 and env.spec.act_dim == 5
    obs = env.reset()
    assert obs.shape == (4, 15) and obs.dtype == np.float32

    action = np.zeros((4, 5), dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs.shape == (4, 15)
    assert reward.shape == (4,) and terminated.shape == (4,) and truncated.shape == (4,)


def test_toy_env_autoreset_at_horizon():
    env = ToyHandEnv(num_envs=1, num_joints=2, episode_steps=5)
    env.reset()
    truncs = []
    for _ in range(5):
        _, _, _, truncated, _ = env.step(np.zeros((1, 2), np.float32))
        truncs.append(bool(truncated[0]))
    assert truncs == [False, False, False, False, True]
    # after auto-reset the internal clock restarts
    assert int(env.t[0]) == 0


def test_factory_builds_toy():
    from worldmodel.common.config import Config

    env = make_env(Config({"backend": "toy", "num_envs": 2, "num_joints": 3,
                           "episode_steps": 8}), seed=1)
    assert env.num_envs == 2 and env.spec.act_dim == 3
