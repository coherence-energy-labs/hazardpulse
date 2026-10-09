"""Coherence fusion: one probability from many forecast sources, weighted by the coherence each has earned.

The coherence framework's law of priced persistence, fitted to forecast combination:
- a source's COHERENCE ENERGY is the information it has cost, E = sum of its log losses on verified cases -- each
  log loss is a Kullback-Leibler cost, the framework's E_coh = kT * D_KL with kT = 1/eta;
- energy is CREATED by each verification and DECAYS with forgetting time tau, so standing must be kept up, not
  inherited -- the operator dE/dt = l(t) - E/tau, in time;
- sources are weighted as a Boltzmann distribution, w ~ exp(-eta * E): low-energy (coherent) sources dominate;
- ALIGNMENT: the pooled log-odds are multiplied by a >= 1, so agreement among sources is amplified rather than
  averaged away.

Mathematically this is prediction with expert advice by exponential weights with discounting, with "sleeping
experts": a source silent on a case is charged the fused forecast's own loss, so absence neither helps nor hurts
it. Deterministic; the same code serves the backtest and live.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

EPS = 0.005            # every probability is read in [EPS, 1 - EPS]: a published 0% means "under half a percent"


@dataclass(frozen=True)
class FusionConfig:
    eta: float = 0.2                 # inverse temperature: how sharply energy separates the sources
    tau_days: float | None = 120.0   # forgetting time; None = never forget (plain Bayesian model averaging at eta=1)
    pool: str = "linear"             # "linear" (a mixture of probabilities) or "log" (a mixture of log-odds)
    align: float = 1.0               # >= 1: amplification of the fused log-odds
    equal: bool = False              # the equal-weight reference: no energies at all


def _clip(p: float) -> float:
    return min(max(float(p), EPS), 1.0 - EPS)


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def log_loss(p: float, y: int) -> float:
    p = _clip(p)
    return -math.log(p if y else 1.0 - p)


@dataclass
class CoherenceFusion:
    cfg: FusionConfig
    energy: dict[str, float] = field(default_factory=dict)
    t_days: float | None = None      # the time the energies are current to (days, any epoch)

    def _decay_to(self, t_days: float) -> None:
        if self.t_days is not None and self.cfg.tau_days is not None and t_days > self.t_days:
            g = math.exp(-(t_days - self.t_days) / self.cfg.tau_days)
            for k in self.energy:
                self.energy[k] *= g
        if self.t_days is None or t_days > self.t_days:
            self.t_days = t_days

    def weights(self, t_days: float, awake: list[str]) -> dict[str, float]:
        """Boltzmann weights over the sources with a forecast now (a new source enters at the lowest energy among
        them -- it has cost nothing yet, and earns or loses standing from its first verification)."""
        if not awake:
            return {}
        if self.cfg.equal:
            return {m: 1.0 / len(awake) for m in awake}
        self._decay_to(t_days)
        known = [self.energy[m] for m in awake if m in self.energy]
        floor = min(known) if known else 0.0
        e = {m: self.energy.get(m, floor) for m in awake}
        lo = min(e.values())
        raw = {m: math.exp(-self.cfg.eta * (v - lo)) for m, v in e.items()}
        z = sum(raw.values())
        return {m: v / z for m, v in raw.items()}

    def fuse(self, t_days: float, probs: Mapping[str, float]) -> tuple[float | None, dict[str, float]]:
        """The fused probability at time ``t_days`` from the sources present (finite values), and the weights."""
        awake = sorted(m for m, p in probs.items() if p is not None and math.isfinite(p))
        if not awake:
            return None, {}
        w = self.weights(t_days, awake)
        if self.cfg.pool == "log":
            x = sum(w[m] * _logit(probs[m]) for m in awake)
        else:
            x = _logit(sum(w[m] * _clip(probs[m]) for m in awake))
        return _sigmoid(self.cfg.align * x), w

    def verify(self, t_days: float, probs: Mapping[str, float], fused: float, y: int, sources: list[str]) -> None:
        """Learn one case that has verified by ``t_days``: each source present at issue pays its own log loss;
        each source in ``sources`` that was silent pays the fused forecast's (sleeping experts)."""
        if self.cfg.equal:
            return
        self._decay_to(t_days)
        silent = log_loss(fused, y)
        for m in sources:
            p = probs.get(m)
            loss = log_loss(p, y) if p is not None and math.isfinite(p) else silent
            self.energy[m] = self.energy.get(m, 0.0) + loss


def run(cases: list[dict], cfg: FusionConfig, sources: list[str], verify_after_days: float) -> list[float | None]:
    """Causal pass over ``cases`` (each ``{"t": days, "probs": {source: p}, "y": 0/1}``) in time order: at each
    time, every case whose outcome is known (issue + ``verify_after_days`` <= now) is learned first, then the
    cases issued now are fused. Returns the fused probability per case, in the input order."""
    order = sorted(range(len(cases)), key=lambda i: cases[i]["t"])
    fusion = CoherenceFusion(cfg)
    fused: list[float | None] = [None] * len(cases)
    pending: list[int] = []
    for i in order:
        now = cases[i]["t"]
        still = []
        for j in pending:
            if cases[j]["t"] + verify_after_days <= now:
                if fused[j] is not None:
                    fusion.verify(cases[j]["t"] + verify_after_days, cases[j]["probs"], fused[j], cases[j]["y"], sources)
            else:
                still.append(j)
        pending = still
        fused[i], _ = fusion.fuse(now, cases[i]["probs"])
        pending.append(i)
    return fused
