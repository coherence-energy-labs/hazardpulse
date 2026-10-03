#!/usr/bin/env python3
"""Program step 1: fit candidates A (long-term smoothed seismicity) and B (A + ETAS-style
short-term clustering) on the FIT split by maximum Bernoulli likelihood, then write their
forecasts for every split (docs/EARTHQUAKE_FORECAST_PROGRAM.md sections 5-6).

Likelihood of P = 1 - exp(-lambda) over every FIT cell-time:
    LL = sum_pos [log(1 - exp(-lambda)) + lambda] - sum_all lambda
(the negatives enter only through the total rate, which the model gives in closed form),
so a fit touches the rate at positive cell-times and the per-issue total, never the full
11,700-cell map. The analytic gradient is checked against central differences before
every optimisation; a mismatch aborts the run.

    python scripts/earthquake_program/fit_ab.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from hazardpulse.earthquake import operational_forecast as of  # noqa: E402

D_GRID_A = [10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 150.0]
D0_GRID_B = [5.0, 10.0, 20.0, 40.0]
LN10 = np.log(10.0)
H = of.HORIZON_DAYS


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# A
# ---------------------------------------------------------------------------

def fit_long_term(s_pos: np.ndarray, u_pos: np.ndarray, n_issue: int) -> tuple[dict, float]:
    def nll(theta):
        mu, eps = np.exp(theta[0]), _sig(theta[1])
        lam = mu * ((1 - eps) * s_pos + eps * u_pos)
        one_m = -np.expm1(-lam)
        ll = np.sum(np.log(one_m) + lam) - n_issue * mu
        f = 1.0 / one_m
        g0 = np.sum(f * lam) - n_issue * mu
        g1 = np.sum(f * mu * (u_pos - s_pos) * eps * (1 - eps))
        return -ll, -np.array([g0, g1])

    best = None
    for x0 in ([np.log(10.0), -3.0], [np.log(5.0), -6.0], [np.log(20.0), -1.0]):
        r = optimize.minimize(nll, np.array(x0), jac=True, method="L-BFGS-B",
                              bounds=[(-10, 10), (-25, 25)])
        if best is None or r.fun < best.fun:
            best = r
    mu, eps = float(np.exp(best.x[0])), float(_sig(best.x[1]))
    return {"mu": mu, "eps": eps, "converged": bool(best.success)}, -float(best.fun)


def long_term_densities(engine: of.RateEngine, issue: np.ndarray) -> np.ndarray:
    return np.stack([engine.long_term_density(t) for t in issue])


# ---------------------------------------------------------------------------
# B
# ---------------------------------------------------------------------------

def omega_and_derivs(a: np.ndarray, p: float):
    """Omega = int_a^{a+H} s^-p ds (a = delta + c) and its derivatives in c and p."""
    b = a + H
    q = 1.0 - p
    a_mp = a ** (-p)
    b_mp = b ** (-p)
    d_dc = b_mp - a_mp
    if abs(q) > 1e-3:
        aq = a * a_mp          # a^q
        bq = b * b_mp
        om = (bq - aq) / q
        la, lb = np.log(a), np.log(b)
        d_dq = ((bq * lb - aq * la) * q - (bq - aq)) / (q * q)
    else:
        la, lb = np.log(a), np.log(b)
        om = (lb - la) + q * (lb ** 2 - la ** 2) / 2 + q * q * (lb ** 3 - la ** 3) / 6
        d_dq = (lb ** 2 - la ** 2) / 2 + q * (lb ** 3 - la ** 3) / 3 + q * q * (lb ** 4 - la ** 4) / 8
    return om, d_dc, -d_dq


class ShortTermLikelihood:
    """Bernoulli log-likelihood of model B on FIT, with its analytic gradient."""

    def __init__(self, engine: of.RateEngine, issue: np.ndarray, pos_k: np.ndarray, pos_c: np.ndarray,
                 s_pos: np.ndarray, u_pos: np.ndarray) -> None:
        ev = engine.events
        self.n_issue = issue.size
        self.s_pos, self.u_pos = s_pos, u_pos
        self.n_pos = pos_k.size
        n_before = np.searchsorted(ev.t, issue, side="left")
        G = engine.G_short.tocsc()
        G.sort_indices()
        pj, pr, pg = [], [], []
        for j, (k, c) in enumerate(zip(pos_k, pos_c)):
            lo, hi = G.indptr[c], G.indptr[c + 1]
            rows = G.indices[lo:hi]
            cut = np.searchsorted(rows, n_before[k], side="left")      # rows are time-ordered events
            if cut:
                pj.append(np.full(cut, j, dtype=np.int32))
                pr.append(rows[:cut].astype(np.int32))
                pg.append(G.data[lo:lo + cut])
        self.pj = np.concatenate(pj)
        pr = np.concatenate(pr)
        self.pg = np.concatenate(pg).astype(np.float32)
        self.pdelta = (issue[pos_k][self.pj] - ev.t[pr]) / of.SEC_DAY
        self.pm5 = (ev.mag[pr] - 5.0).astype(np.float32)
        self.n_pairs = self.pj.size
        # totals: every event before each issue time, times its in-domain kernel mass
        # (sliced on the fly: storing per-issue copies would cost ~1 GB)
        self.ev_t = ev.t
        self.ev_m5 = ev.mag - 5.0
        self.mass = engine.short_mass
        self.issue = issue
        self.n_before = n_before

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        logmu, logit_eps, logK, alpha, logc, p = theta
        mu, eps, K, c = np.exp(logmu), _sig(logit_eps), np.exp(logK), np.exp(logc)
        b = mu * ((1 - eps) * self.s_pos + eps * self.u_pos)
        om, om_c, om_p = omega_and_derivs(self.pdelta + c, p)
        prod = 10.0 ** (alpha * self.pm5) * self.pg
        term = K * prod * om
        npos = self.n_pos
        lam_st = np.bincount(self.pj, weights=term, minlength=npos)
        lam = b + lam_st
        one_m = -np.expm1(-lam)
        f = 1.0 / one_m
        ll = np.sum(np.log(one_m) + lam)
        g = np.zeros(6)
        g[0] = np.sum(f * b)
        g[1] = np.sum(f * mu * (self.u_pos - self.s_pos) * eps * (1 - eps))
        g[2] = np.sum(f * lam_st)
        g[3] = np.sum(f * np.bincount(self.pj, weights=term * LN10 * self.pm5, minlength=npos))
        g[4] = np.sum(f * np.bincount(self.pj, weights=K * prod * om_c * c, minlength=npos))
        g[5] = np.sum(f * np.bincount(self.pj, weights=K * prod * om_p, minlength=npos))
        # minus the total rate
        tot = self.n_issue * mu
        gt = np.zeros(6)
        gt[0] = self.n_issue * mu
        for k in range(self.n_issue):
            n = int(self.n_before[k])
            if n == 0:
                continue
            dlt = (self.issue[k] - self.ev_t[:n]) / of.SEC_DAY
            m5 = self.ev_m5[:n]
            mass = self.mass[:n]
            o, oc, op = omega_and_derivs(dlt + c, p)
            w = K * 10.0 ** (alpha * m5) * mass
            wo = w * o
            s = wo.sum()
            tot += s
            gt[2] += s
            gt[3] += np.sum(wo * LN10 * m5)
            gt[4] += np.sum(w * oc) * c
            gt[5] += np.sum(w * op)
        ll -= tot
        g -= gt
        return -float(ll), -g


def check_gradient(fun, theta, h=1e-5) -> float:
    _, g = fun(theta)
    worst = 0.0
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = h
        num = (fun(theta + e)[0] - fun(theta - e)[0]) / (2 * h)
        worst = max(worst, abs(num - g[i]) / max(1.0, abs(num)))
    return worst


def fit_short_term(lik: ShortTermLikelihood, long_params: dict) -> tuple[dict, float, float]:
    bounds = [(-10, 10), (-25, 25), (-30, 5), (0.0, 3.0), (np.log(1e-4), np.log(10.0)), (0.5, 3.0)]
    starts = [
        [np.log(long_params["mu"] * 0.7), np.log(long_params["eps"] / (1 - long_params["eps"])), np.log(1e-3), 1.0, np.log(0.01), 1.1],
        [np.log(long_params["mu"] * 0.5), np.log(long_params["eps"] / (1 - long_params["eps"])), np.log(1e-4), 1.6, np.log(0.1), 1.3],
    ]
    gcheck = check_gradient(lik, np.array(starts[0], dtype=float))
    if gcheck > 1e-4:
        raise RuntimeError(f"analytic gradient disagrees with central differences (rel {gcheck:.2e})")
    best = None
    for x0 in starts:
        r = optimize.minimize(lik, np.array(x0, dtype=float), jac=True, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    params = {"mu": float(np.exp(x[0])), "eps": float(_sig(x[1])), "K": float(np.exp(x[2])),
              "alpha": float(x[3]), "c_days": float(np.exp(x[4])), "p": float(x[5]),
              "converged": bool(best.success), "message": str(best.message), "n_iter": int(best.nit)}
    return params, -float(best.fun), gcheck


def main() -> int:
    t_start = time.time()
    events = C.load_events(4.5)
    issue = C.all_issue_times()
    split = np.array([C.split_of(t) for t in issue])
    fit = split == "fit"
    t_fit = issue[fit]
    Y = C.binary_targets(events, issue)
    Yf = Y[fit]
    pos_k, pos_c = np.nonzero(Yf)
    n_fit = int(t_fit.size)
    area_pdf = of.cell_areas() / of.cell_areas().sum()
    u_pos = area_pdf[pos_c]
    p0 = float(Yf.mean())
    ll0 = float(np.sum(np.where(Yf, np.log(p0), np.log1p(-p0))))
    print(f"FIT: {n_fit} issue times, {pos_k.size} positive cell-windows, p0 = {p0:.3e}, LL0 = {ll0:.1f}")
    report: dict = {"fit": {"n_issue": n_fit, "n_positive": int(pos_k.size), "p0": p0, "ll_reference": ll0},
                    "A_grid": [], "B_grid": []}
    prior: dict = {}
    if "--resume" in sys.argv and (C.RESULTS / "fit_ab.json").exists():
        # same deterministic computation; rows already recorded are reused, not re-fitted
        prior = json.loads((C.RESULTS / "fit_ab.json").read_text(encoding="utf-8"))
        if prior.get("fit") != report["fit"]:
            raise RuntimeError("--resume: the recorded fit_ab.json describes a different FIT population")
        report["resumed_from_partial_run"] = True

    # ---- A: grid over (input, width), (mu, eps) by likelihood
    best_a = None
    done_a = {(r["kernel_km"], r["declustered"]): r for r in prior.get("A_grid", [])}
    for declustered in (False, True):
        for d in D_GRID_A:
            if (d, declustered) in done_a:
                row = done_a[(d, declustered)]
                report["A_grid"].append(row)
                if best_a is None or row["ll"] > best_a[0]:
                    best_a = (row["ll"], row, None)
                continue
            t0 = time.time()
            eng = of.RateEngine(events, kernel_km=d, declustered=declustered)
            S = long_term_densities(eng, t_fit)
            params, ll = fit_long_term(S[pos_k, pos_c], u_pos, n_fit)
            row = {"kernel_km": d, "declustered": declustered, **params, "ll": ll,
                   "ig_per_target": (ll - ll0) / pos_k.size, "seconds": round(time.time() - t0, 1)}
            report["A_grid"].append(row)
            print(f"  A d={d:5.0f} decl={declustered!s:5}: LL {ll:.2f}  IG {row['ig_per_target']:.4f}  "
                  f"mu {params['mu']:.3f} eps {params['eps']:.4f}  ({row['seconds']}s)")
            if best_a is None or ll > best_a[0]:
                best_a = (ll, row, S)
            C.write_json("fit_ab.json", report)
    ll_a, row_a, S_fit = best_a
    if S_fit is None:   # resumed: recompute the chosen map and check it reproduces the recorded fit
        eng = of.RateEngine(events, kernel_km=row_a["kernel_km"], declustered=row_a["declustered"])
        S_fit = long_term_densities(eng, t_fit)
        params, ll = fit_long_term(S_fit[pos_k, pos_c], u_pos, n_fit)
        if abs(ll - ll_a) > 1e-6 * abs(ll_a):
            raise RuntimeError(f"--resume: recomputed A likelihood {ll} != recorded {ll_a}")
        del eng
    spec_a = of.ModelSpec("A", of.LongTermParams(kernel_km=row_a["kernel_km"], declustered=row_a["declustered"],
                                                 mu=row_a["mu"], eps=row_a["eps"]))
    report["A_chosen"] = spec_a.to_dict()
    print(f"A chosen: {spec_a.to_dict()}")

    # ---- B: A's map + short-term term, d0 by likelihood
    s_pos = S_fit[pos_k, pos_c]
    best_b = None
    done_b = {r["d0_km"]: r for r in prior.get("B_grid", [])}
    for d0 in D0_GRID_B:
        if d0 in done_b:
            row = done_b[d0]
            report["B_grid"].append(row)
            if best_b is None or row["ll"] > best_b[0]:
                best_b = (row["ll"], row)
            continue
        t0 = time.time()
        eng = of.RateEngine(events, kernel_km=row_a["kernel_km"], declustered=row_a["declustered"], d0_km=d0)
        lik = ShortTermLikelihood(eng, t_fit, pos_k, pos_c, s_pos, u_pos)
        params, ll, gcheck = fit_short_term(lik, row_a)
        row = {"d0_km": d0, **params, "ll": ll, "ig_per_target": (ll - ll0) / pos_k.size,
               "gradient_check_rel_err": gcheck, "n_pairs": int(lik.n_pairs), "seconds": round(time.time() - t0, 1)}
        report["B_grid"].append(row)
        print(f"  B d0={d0:4.0f}: LL {ll:.2f}  IG {row['ig_per_target']:.4f}  K {params['K']:.3e} "
              f"alpha {params['alpha']:.3f} c {params['c_days']:.4f} p {params['p']:.3f} mu {params['mu']:.3f} "
              f"eps {params['eps']:.4f} (grad err {gcheck:.1e}, {row['seconds']}s)")
        if best_b is None or ll > best_b[0]:
            best_b = (ll, row)
        C.write_json("fit_ab.json", report)
        del lik, eng
    ll_b, row_b = best_b
    spec_b = of.ModelSpec("B", of.LongTermParams(kernel_km=row_a["kernel_km"], declustered=row_a["declustered"],
                                                 mu=row_b["mu"], eps=row_b["eps"]),
                          of.ShortTermParams(d0_km=row_b["d0_km"], K=row_b["K"], alpha=row_b["alpha"],
                                             c_days=row_b["c_days"], p=row_b["p"]))
    report["B_chosen"] = spec_b.to_dict()
    print(f"B chosen: {spec_b.to_dict()}")
    C.write_json("fit_ab.json", report)

    # ---- forecasts for every issue time; save per split, plus the maps C reads
    t0 = time.time()
    eng = of.RateEngine(events, spec_b)
    P_a = np.empty((issue.size, of.N_CELLS))
    P_b = np.empty((issue.size, of.N_CELLS))
    lam_a = np.empty((issue.size, of.N_CELLS), dtype=np.float32)
    lam_st = np.empty((issue.size, of.N_CELLS), dtype=np.float32)
    for k, t in enumerate(issue):
        s = eng.long_term_density(t)
        la = spec_a.long.mu * ((1 - spec_a.long.eps) * s + spec_a.long.eps * eng.area_pdf)
        r = eng.rates(t, spec_b)
        P_a[k] = -np.expm1(-la)
        P_b[k] = r["probability"]
        lam_a[k] = la
        lam_st[k] = r["lambda_short"]
    for name in ("fit", "choose", "dev", "final"):
        sel = split == name
        C.save_pred("A", name, P_a[sel])
        C.save_pred("B", name, P_b[sel])
    np.save(C.PROGRAM_CACHE / "maps_issue_times.npy", issue)
    np.save(C.PROGRAM_CACHE / "maps_lambda_A.npy", lam_a)
    np.save(C.PROGRAM_CACHE / "maps_lambda_short_B.npy", lam_st)
    # in-sample check: FIT likelihood recomputed from the saved full maps equals the fitted one
    for cand, P, ll_fit in (("A", P_a[fit], ll_a), ("B", P_b[fit], ll_b)):
        llm = float(np.sum(np.where(Yf, np.log(np.clip(P, 1e-300, 1)), np.log1p(-P))))
        report[f"{cand}_full_map_fit_ll"] = llm
        print(f"  {cand}: FIT LL from full maps {llm:.2f} vs fit {ll_fit:.2f}")
        if abs(llm - ll_fit) > 1e-6 * abs(ll_fit) + 1e-3:
            raise RuntimeError(f"{cand}: full-map likelihood {llm} != fitted likelihood {ll_fit}")
    report["maps_seconds"] = round(time.time() - t0, 1)
    report["total_seconds"] = round(time.time() - t_start, 1)
    C.write_json("fit_ab.json", report)
    print(f"done in {report['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
