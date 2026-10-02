"""Tie-aware ranking metrics shared by every hazard's verification code.

Why this module exists
----------------------
The prospective scorers used to compute ROC-AUC by sorting with an unstable
``np.argsort`` and integrating one trapezoid PER SAMPLE. Inside a block of tied
scores that walk credits each positive according to wherever the sort happened to
place it, so the answer depends on row order, not on the forecast. HazardPulse
forecasts are full of ties -- e.g. ~11,600 inactive earthquake grid cells all at
``default_probability = 0.0`` -- so the published earthquake mean AUC was inflated
(0.6972 published vs 0.6564 tie-aware over the same 600 forecasts), and a constant
scorer could score 1.0 or 0.0 depending on how the rows were ordered.

Both metrics here are functions of the multiset of (score, label) pairs only:
permuting the rows can never change the result.

* :func:`roc_auc` -- the Mann-Whitney probability P(s+ > s-) + 1/2 P(s+ == s-),
  i.e. the trapezoid of the ROC curve drawn with ONE vertex per distinct score.
* :func:`average_precision` -- step-wise area under the precision-recall curve
  with one point per distinct score (the estimator used by scikit-learn's
  ``average_precision_score``). Linear (trapezoidal) interpolation in PR space is
  not used because it is optimistic (Davis & Goadrich, ICML 2006).
"""

from __future__ import annotations

import numpy as np

__all__ = ["roc_auc", "average_precision"]


def _validated(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true and y_score lengths differ ({y.size} vs {s.size})")
    if y.size and not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y_true must be binary (0/1)")
    if np.any(np.isnan(s)):
        raise ValueError("y_score contains NaN; a NaN has no rank")
    return y, s


def _grouped_counts(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positives / negatives per distinct score, ordered from the highest score down."""
    uniq, inv = np.unique(s, return_inverse=True)            # ascending
    pos = np.bincount(inv, weights=y, minlength=uniq.size)[::-1]
    tot = np.bincount(inv, minlength=uniq.size).astype(np.float64)[::-1]
    return pos, tot - pos


def roc_auc(y_true, y_score) -> float:
    """Tie-aware ROC-AUC (ties count 1/2). NaN when only one class is present."""
    y, s = _validated(y_true, y_score)
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    pos, neg = _grouped_counts(y, s)
    neg_above = np.concatenate(([0.0], np.cumsum(neg)[:-1]))  # negatives scored strictly higher
    # Each positive beats every negative strictly below it and ties half of its own block.
    wins = float(np.sum(pos * (n_neg - neg_above - neg) + 0.5 * pos * neg))
    return wins / (n_pos * n_neg)


def average_precision(y_true, y_score) -> float:
    """Tie-aware average precision. NaN when there are no positives."""
    y, s = _validated(y_true, y_score)
    n_pos = float(y.sum())
    if n_pos == 0.0:
        return float("nan")
    pos, neg = _grouped_counts(y, s)
    tp = np.cumsum(pos)
    fp = np.cumsum(neg)
    precision = tp / (tp + fp)
    recall_step = pos / n_pos
    return float(np.sum(recall_step * precision))
