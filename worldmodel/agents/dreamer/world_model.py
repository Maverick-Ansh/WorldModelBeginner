"""DreamerV3 world model: encoder + RSSM + decoder/reward/continue heads, and its loss."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...common.distributions import SymlogMSE, SymlogTwoHot
from ...common.nets import VectorDecoder, VectorEncoder, build_mlp, zero_init_
from .rssm import RSSM


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, cfg):
        super().__init__()
        hidden = cfg.hidden
        embed_dim = cfg.embed_dim
        layers = cfg.mlp_layers
        self.num_bins = cfg.num_bins

        self.encoder = VectorEncoder(obs_dim, hidden, embed_dim, layers)
        self.rssm = RSSM(embed_dim, action_dim, cfg.deter, cfg.groups, cfg.classes, hidden,
                         unimix=cfg.unimix)
        feat = self.rssm.feat_dim
        self.decoder = VectorDecoder(feat, hidden, obs_dim, layers)
        self.reward_head = build_mlp(feat, hidden, self.num_bins, layers)
        self.cont_head = build_mlp(feat, hidden, 1, layers)
        zero_init_(self.reward_head)
        zero_init_(self.cont_head)

        self.free = cfg.kl_free
        self.dyn_scale = cfg.kl_dyn
        self.rep_scale = cfg.kl_rep

    # -- prediction heads -------------------------------------------------
    def reward_dist(self, feat: Tensor) -> SymlogTwoHot:
        return SymlogTwoHot(self.reward_head(feat))

    def cont_dist(self, feat: Tensor) -> torch.distributions.Bernoulli:
        return torch.distributions.Bernoulli(logits=self.cont_head(feat).squeeze(-1))

    def decode_dist(self, feat: Tensor) -> SymlogMSE:
        return SymlogMSE(self.decoder(feat))

    # -- training ---------------------------------------------------------
    def loss(self, batch: dict[str, Tensor]):
        embed = self.encoder(batch["obs"])
        states, post_logits, prior_logits, _ = self.rssm.observe(
            embed, batch["action"], batch["is_first"]
        )
        feat = self.rssm.get_feat(states)

        recon = -self.decode_dist(feat).log_prob(batch["obs"]).mean()
        reward = -self.reward_dist(feat).log_prob(batch["reward"]).mean()
        cont = -self.cont_dist(feat).log_prob(batch["cont"]).mean()
        kl, kl_metrics = self.rssm.kl_loss(
            post_logits, prior_logits, self.free, self.dyn_scale, self.rep_scale
        )
        loss = recon + reward + cont + kl

        metrics = {
            "wm_loss": loss.detach(),
            "wm_recon": recon.detach(),
            "wm_reward": reward.detach(),
            "wm_cont": cont.detach(),
            "wm_kl": kl.detach(),
            **kl_metrics,
        }
        # Detach start states so behavior learning does not backprop into the world model.
        start = {k: v.detach() for k, v in states.items()}
        return loss, start, metrics
