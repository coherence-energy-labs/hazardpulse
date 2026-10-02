"""Pure-numpy serving of the deep earthquake nowcast (bidirectional GRU + attention).

The production scorer (GitHub Actions earthquake-score.yml) installs only numpy+scipy --
no torch. This reimplements the trained ``SeqModel`` forward pass in numpy, matching
PyTorch's GRU gate convention, so the deep model serves without the heavy dependency
(consistent with the existing pure-numpy GBT / VerifiableForest serve paths). Weights are
loaded from the .npz exported by ``scripts/export_deep_eq_serve.py``; verified bit-close
(<1e-5) to the torch model on the held-out test sequences.

The model reads, per location, the K most-recent catalog events within ``radius_km`` and
5 years before the reference time, each encoded as
[log1p(dt_days), mag, dist/radius, depth/700, sin(az), cos(az)] -- exactly as training.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEC_DAY = 86400.0
EARTH_KM = 6371.0
# Every served sequence reads the events of the 5 years before the reference time,
# exactly as training did (scripts/deep_sequence_earthquake.py::_seq_one and
# scripts/deep_operational_earthquake.py::_build_one: ``t0 = ref - 5 * 365 * SEC_DAY``).
# The catalog handed to these scorers must therefore cover at least this many days,
# or each sequence is silently cut short relative to what the weights were fit on.
SEQUENCE_LOOKBACK_DAYS = 5 * 365


def _sigmoid(x):
    x = np.asarray(x, float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _hav_az(lat, lon, lats, lons):
    """Distance (km) + azimuth (rad) -- matches the training _haversine in deep_sequence."""
    rlat, rlon = np.radians(lat), np.radians(lon)
    rla, rlo = np.radians(lats), np.radians(lons)
    dlon = rlo - rlon
    d = (np.sin((rla - rlat) / 2) ** 2
         + np.cos(rlat) * np.cos(rla) * np.sin(dlon / 2) ** 2)
    dist = EARTH_KM * 2 * np.arcsin(np.sqrt(d))
    az = np.arctan2(np.sin(dlon) * np.cos(rla),
                    np.cos(rlat) * np.sin(rla) - np.sin(rlat) * np.cos(rla) * np.cos(dlon))
    return dist, az


class DeepEQScorer:
    """Loads exported weights + serves P(M5+ within radius/365d) for a location/time."""

    def __init__(self, npz_path, calib_path=None):
        d = np.load(npz_path)
        self.Wp = d["proj_w"]; self.bp = d["proj_b"]
        self.Wih = d["gru_ih"]; self.Whh = d["gru_hh"]; self.bih = d["gru_bih"]; self.bhh = d["gru_bhh"]
        self.Wih_r = d["gru_ih_r"]; self.Whh_r = d["gru_hh_r"]
        self.bih_r = d["gru_bih_r"]; self.bhh_r = d["gru_bhh_r"]
        self.Wa = d["att_w"]; self.ba = d["att_b"]
        self.Wh1 = d["head_w1"]; self.bh1 = d["head_b1"]
        self.Wh2 = d["head_w2"]; self.bh2 = d["head_b2"]
        self.mu = d["norm_mu"].astype(np.float64); self.sd = d["norm_sd"].astype(np.float64)
        self.K = int(d["K"]); self.radius_km = float(d["radius_km"])
        self.H = self.Wp.shape[0]
        self.calib = None
        if calib_path and Path(calib_path).exists():
            self.calib = json.loads(Path(calib_path).read_text(encoding="utf-8"))

    # --- one GRU direction (PyTorch convention: gates [r, z, n]) ---
    def _gru_dir(self, X, Wih, Whh, bih, bhh, reverse):
        K = X.shape[0]; H = self.H
        h = np.zeros(H); out = np.zeros((K, H))
        for t in (range(K - 1, -1, -1) if reverse else range(K)):
            gi = Wih @ X[t] + bih
            gh = Whh @ h + bhh
            r = _sigmoid(gi[:H] + gh[:H])
            z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h = (1.0 - z) * n + z * h
            out[t] = h
        return out

    def forward(self, X, m):
        """X: (K,6) raw features, m: (K,) mask -> probability (calibrated if available)."""
        Xn = (np.asarray(X, float) - self.mu) / self.sd
        p = np.maximum(0.0, Xn @ self.Wp.T + self.bp)          # proj + relu
        hf = self._gru_dir(p, self.Wih, self.Whh, self.bih, self.bhh, reverse=False)
        hb = self._gru_dir(p, self.Wih_r, self.Whh_r, self.bih_r, self.bhh_r, reverse=True)
        z = np.concatenate([hf, hb], axis=1)                   # (K, 2H)
        a = (z @ self.Wa.T + self.ba).reshape(-1)              # (K,)
        a = np.where(np.asarray(m) == 0, -1e9, a)
        a = a - a.max(); e = np.exp(a); a = e / e.sum()
        pooled = (z * a[:, None]).sum(0)                       # (2H,)
        hh = np.maximum(0.0, self.Wh1 @ pooled + self.bh1)
        logit = float((self.Wh2 @ hh + self.bh2).reshape(()))
        return self._apply_calib(_sigmoid(logit))

    def _apply_calib(self, p):
        if not self.calib:
            return float(p)
        if self.calib.get("method") == "temp":
            T = float(self.calib.get("temperature", 1.0))
            logit = np.log(p / (1 - p + 1e-12) + 1e-12)
            return float(_sigmoid(logit / T))
        if self.calib.get("method") == "iso":
            x = np.asarray(self.calib["iso_x"]); y = np.asarray(self.calib["iso_y"])
            return float(np.interp(p, x, y))
        return float(p)

    def build_sequence(self, cat, lat, lon, ref_epoch):
        """Build the (K,6) causal sequence + mask from a CatalogArrays-like object
        (attributes times/lats/lons/mags/depths as numpy arrays). Mirrors training _seq_one."""
        K, R = self.K, self.radius_km
        X = np.zeros((K, 6), np.float64); m = np.zeros(K, np.float64)
        t0 = ref_epoch - SEQUENCE_LOOKBACK_DAYS * SEC_DAY
        sel = ((cat.times >= t0) & (cat.times < ref_epoch)
               & (np.abs(cat.lats - lat) < 6) & (np.abs(cat.lons - lon) < 6))
        idx = np.where(sel)[0]
        if idx.size:
            dist, az = _hav_az(lat, lon, cat.lats[idx], cat.lons[idx])
            near = dist < R
            idx, dist, az = idx[near], dist[near], az[near]
            if idx.size:
                order = np.argsort(cat.times[idx])[-K:]
                idx, dist, az = idx[order], dist[order], az[order]
                dd = (ref_epoch - cat.times[idx]) / SEC_DAY
                seq = np.stack([np.log1p(dd), cat.mags[idx], dist / R,
                                np.clip(cat.depths[idx], 0, 700) / 700.0,
                                np.sin(az), np.cos(az)], axis=1)
                X[K - len(idx):] = seq; m[K - len(idx):] = 1.0
        return X, m

    def score(self, cat, lat, lon, ref_epoch):
        """Returns probability, or None if no causal events in the window."""
        X, m = self.build_sequence(cat, lat, lon, ref_epoch)
        if m.sum() == 0:
            return None
        return self.forward(X, m)


class OperationalEQScorer(DeepEQScorer):
    """Operational forecaster: ranks "which active cell ruptures next" (real WHERE-skill,
    beats climatology). Input adds 3 cross-location context channels (abs lat/lon + recent
    rate) to the 6 per-event channels -- d=9. Forward is inherited (dim-agnostic)."""

    def build_sequence(self, cat, lat, lon, ref_epoch):
        if self.Wp.shape[1] >= 20:
            return self._build_multiscale(cat, lat, lon, ref_epoch)
        K, R = self.K, self.radius_km
        X = np.zeros((K, 9), np.float64); m = np.zeros(K, np.float64)
        t0 = ref_epoch - SEQUENCE_LOOKBACK_DAYS * SEC_DAY
        sel = ((cat.times >= t0) & (cat.times < ref_epoch)
               & (np.abs(cat.lats - lat) < 6) & (np.abs(cat.lons - lon) < 6))
        idx = np.where(sel)[0]
        if idx.size:
            dist, az = _hav_az(lat, lon, cat.lats[idx], cat.lons[idx])
            near = dist < R
            idx, dist, az = idx[near], dist[near], az[near]
            if idx.size:
                order = np.argsort(cat.times[idx])[-K:]
                idx, dist, az = idx[order], dist[order], az[order]
                dd = (ref_epoch - cat.times[idx]) / SEC_DAY
                n_1yr = float((dd < 365).sum())
                loc = [lat / 90.0, lon / 180.0, np.log1p(n_1yr) / 6.0]
                seq = np.stack([np.log1p(dd), cat.mags[idx], dist / R,
                                np.clip(cat.depths[idx], 0, 700) / 700.0, np.sin(az), np.cos(az)]
                               + [np.full(len(idx), c) for c in loc], axis=1)
                X[K - len(idx):] = seq; m[K - len(idx):] = 1.0
        return X, m

    def _build_multiscale(self, cat, lat, lon, ref):
        """20-channel multi-scale build (must MATCH deep_operational_earthquake._build_one v3):
        6 per-event + lat/lon + per-radius[100,300,500]{recent, baseline, anomaly} + maxmag + 2 stress."""
        K, R = self.K, self.radius_km
        X = np.zeros((K, 20), np.float64); m = np.zeros(K, np.float64)
        bsel = ((cat.times < ref) & (np.abs(cat.lats - lat) < 10) & (np.abs(cat.lons - lon) < 10))
        bi = np.where(bsel)[0]
        if bi.size == 0:
            return X, m
        bd = _hav_az(lat, lon, cat.lats[bi], cat.lons[bi])[0]
        bdays = (ref - cat.times[bi]) / SEC_DAY
        ctx = [lat / 90.0, lon / 180.0]
        for rad, rec_n, base_n in [(100, 5.0, 8.0), (300, 6.0, 9.0), (500, 7.0, 10.0)]:
            inr = bd < rad
            rec = float(((bdays < 180) & inr).sum())
            base = float(((bdays >= 365) & (bdays < 5 * 365) & inr).sum()) / 4.0 / 2.0
            ctx.append(np.log1p(rec) / rec_n)
            ctx.append(np.log1p(((bdays < 5 * 365) & inr).sum()) / base_n)
            ctx.append(np.clip(np.log1p(rec) - np.log1p(base), -3, 3) / 3.0)
        maxmag = float(cat.mags[bi[bd < 300]].max()) if (bd < 300).any() else 0.0
        ctx.append(maxmag / 9.0)
        big = (bd < 1000) & (cat.mags[bi] >= 6.5) & (bdays < 2 * 365)
        if big.any():
            nb = np.argmin(bdays[big]); ctx.append((1000 - bd[big][nb]) / 1000.0)
            ctx.append(max(0.0, 1 - bdays[big][nb] / 730.0))
        else:
            ctx.append(0.0); ctx.append(0.0)
        t0 = ref - SEQUENCE_LOOKBACK_DAYS * SEC_DAY
        sel = ((cat.times >= t0) & (cat.times < ref)
               & (np.abs(cat.lats - lat) < 6) & (np.abs(cat.lons - lon) < 6))
        idx = np.where(sel)[0]
        if idx.size:
            dist, az = _hav_az(lat, lon, cat.lats[idx], cat.lons[idx])
            near = dist < R
            idx, dist, az = idx[near], dist[near], az[near]
        if idx.size:
            order = np.argsort(cat.times[idx])[-K:]
            idx, dist, az = idx[order], dist[order], az[order]
            dd = (ref - cat.times[idx]) / SEC_DAY
            seq = np.stack([np.log1p(dd), cat.mags[idx], dist / R,
                            np.clip(cat.depths[idx], 0, 700) / 700.0, np.sin(az), np.cos(az)]
                           + [np.full(len(idx), c) for c in ctx], axis=1)
            X[K - len(idx):] = seq; m[K - len(idx):] = 1.0
        else:
            X[K - 1] = np.array([0, 0, 0, 0, 0, 0] + ctx, np.float64); m[K - 1] = 1.0
        return X, m


def load_deep_eq_scorer(npz_path, calib_path=None):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        return None
    z = np.load(npz_path)
    kind = str(z["kind"]) if "kind" in z else "nowcast"
    cls = OperationalEQScorer if kind == "operational" else DeepEQScorer
    return cls(npz_path, calib_path)
