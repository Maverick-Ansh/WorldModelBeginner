from pathlib import Path

from worldmodel.common.config import load_configs

ROOT = Path(__file__).resolve().parent.parent


def test_merge_and_override_types():
    cfg = load_configs(
        [ROOT / "configs/default.yaml", ROOT / "configs/algo/dreamerv3.yaml",
         ROOT / "configs/env/toy.yaml"],
        overrides=["seed=5", "algo.horizon=7", "algo.gamma=0.9", "env.num_envs=3",
                   "device=cpu"],
    )
    assert cfg.seed == 5 and isinstance(cfg.seed, int)
    assert cfg.algo.name == "dreamerv3"          # from algo file
    assert cfg.algo.horizon == 7                  # overridden int
    assert isinstance(cfg.algo.gamma, float) and cfg.algo.gamma == 0.9
    assert cfg.env.backend == "toy"               # untouched string
    assert cfg.env.num_envs == 3
    # nested attribute and item access agree
    assert cfg["algo"]["deter"] == cfg.algo.deter


def test_flat_for_logging():
    cfg = load_configs([ROOT / "configs/default.yaml"], overrides=["seed=1"])
    flat = cfg.flat()
    assert flat["seed"] == 1
    assert "total_steps" in flat
