import torch

from worldmodel.common.distributions import (
    SymlogTwoHot,
    TanhNormal,
    categorical_kl,
    symexp,
    symlog,
)


def test_symlog_roundtrip():
    x = torch.tensor([-1000.0, -3.0, 0.0, 2.5, 1e4])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-3, rtol=1e-4)


def test_two_hot_recovers_value():
    base = SymlogTwoHot(torch.zeros(2, 64))
    target = torch.tensor([3.0, -7.5])
    # logits whose softmax equals the two-hot exactly -> expectation recovers the target.
    logits = torch.log(base.two_hot(target).clamp_min(1e-9))
    dist = SymlogTwoHot(logits)
    assert torch.allclose(dist.mean, target, atol=1e-2)
    # log_prob is a valid (non-positive) log-likelihood, bounded below by -log(2).
    lp = dist.log_prob(target)
    assert torch.isfinite(lp).all() and (lp <= 1e-5).all() and (lp > -0.8).all()


def test_tanh_normal_bounds_and_logprob():
    mean = torch.zeros(4, 3)
    std = torch.ones(4, 3)
    dist = TanhNormal(mean, std)
    action, logp = dist.rsample()
    assert action.shape == (4, 3) and logp.shape == (4,)
    assert (action.abs() < 1.0).all()
    assert torch.isfinite(logp).all()
    assert (dist.mode.abs() < 1.0).all()


def test_categorical_kl_zero_for_identical():
    logits = torch.randn(5, 8, 6)  # [batch, groups, classes]
    kl = categorical_kl(logits, logits.clone())
    assert kl.shape == (5,)
    assert torch.allclose(kl, torch.zeros(5), atol=1e-5)
