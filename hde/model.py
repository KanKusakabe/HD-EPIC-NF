"""Conditional density over a 3-D kitchen location, conditioned on the kitchen.

FlowHead  -- zuko conditional Neural Spline Flow (captures the multimodal layout).
GMMHead   -- conditional diagonal Gaussian mixture (the baseline we must beat).

SURPRISE = -log p(x,y,z | kitchen).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

DIM = 3


class KitchenEncoder(nn.Module):
    def __init__(self, n_kitchen, emb=16, out_dim=48):
        super().__init__()
        self.emb = nn.Embedding(n_kitchen, emb)
        self.mlp = nn.Sequential(nn.Linear(emb, out_dim), nn.ReLU(),
                                 nn.Linear(out_dim, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, k_idx):
        return self.mlp(self.emb(k_idx))


class FlowHead(nn.Module):
    def __init__(self, ctx_dim, transforms=4, hidden=(96, 96)):
        super().__init__()
        self.flow = zuko.flows.NSF(features=DIM, context=ctx_dim,
                                   transforms=transforms, hidden_features=hidden)

    def log_prob(self, c, y):
        return self.flow(c).log_prob(y)

    def sample(self, c, n=1):
        return self.flow(c).sample((n,)).permute(1, 0, 2)


class GMMHead(nn.Module):
    def __init__(self, ctx_dim, k=10):
        super().__init__()
        self.k = k
        self.net = nn.Linear(ctx_dim, k * (1 + 2 * DIM))

    def _p(self, c):
        o = self.net(c)
        logit = o[:, :self.k]
        mu = o[:, self.k:self.k + self.k * DIM].view(-1, self.k, DIM)
        log_sd = o[:, self.k + self.k * DIM:].view(-1, self.k, DIM).clamp(-6, 3)
        return logit, mu, log_sd

    def log_prob(self, c, y):
        logit, mu, log_sd = self._p(c)
        y = y.unsqueeze(1)
        comp = -0.5 * (((y - mu) / log_sd.exp()) ** 2 + 2 * log_sd
                       + torch.log(torch.tensor(2 * torch.pi))).sum(-1)
        return torch.logsumexp(F.log_softmax(logit, -1) + comp, dim=-1)

    def sample(self, c, n=1):
        logit, mu, log_sd = self._p(c)
        idx = torch.multinomial(F.softmax(logit, -1), n, replacement=True)
        b = torch.arange(c.shape[0]).unsqueeze(1)
        return mu[b, idx] + log_sd[b, idx].exp() * torch.randn(c.shape[0], n, DIM)


class PlacementModel(nn.Module):
    def __init__(self, n_kitchen, head="flow", **kw):
        super().__init__()
        self.encoder = KitchenEncoder(n_kitchen)
        self.head = (FlowHead(self.encoder.out_dim, **kw) if head == "flow"
                     else GMMHead(self.encoder.out_dim, **kw))

    def condition(self, k_idx):
        return self.encoder(k_idx)

    def log_prob(self, k_idx, y):
        return self.head.log_prob(self.condition(k_idx), y)

    def nll(self, k_idx, y):
        return -self.log_prob(k_idx, y).mean()
