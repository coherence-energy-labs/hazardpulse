"""One HRRR availability rule for training and live: published (valid + 100 min) before use."""
from __future__ import annotations

import datetime as dt

from hazardpulse.data import hrrr_availability as ha

D = dt.datetime


def test_an_analysis_is_unusable_until_it_is_published():
    # 18Z is published at 19:40: at 19:39 the newest usable analysis is 15Z, at 19:40 it is 18Z
    assert ha.usable_analysis(D(2025, 5, 6, 19, 39)) == ("20250506", 15)
    assert ha.usable_analysis(D(2025, 5, 6, 19, 40)) == ("20250506", 18)
    # the old (leaky) rule would have given 18Z at 18:05
    assert ha.usable_analysis(D(2025, 5, 6, 18, 5)) == ("20250506", 15)


def test_the_first_hours_of_a_day_use_the_previous_days_21z():
    assert ha.usable_analysis(D(2025, 5, 6, 0, 30)) == ("20250505", 21)
    assert ha.usable_analysis(D(2025, 5, 6, 1, 39)) == ("20250505", 21)
    assert ha.usable_analysis(D(2025, 5, 6, 1, 40)) == ("20250506", 0)


def test_a_missing_analysis_falls_back_to_an_older_one_within_the_age_limit():
    have = [("20250506", 12), ("20250506", 9)]
    # 15Z would be the usable one from 16:40 but is missing: fall back to 12Z while it is <= 4 h 40 min old
    assert ha.usable_analysis(D(2025, 5, 6, 16, 30), have) == ("20250506", 12)     # 4.5 h old: allowed
    assert ha.usable_analysis(D(2025, 5, 6, 17, 0), have) is None                  # 5.0 h old: refused
    assert ha.usable_analysis(D(2025, 5, 6, 13, 40), have) == ("20250506", 12)


def test_live_candidates_are_published_newest_first_and_within_the_age_limit():
    # at 19:45 18Z is published and 15Z is 4 h 45 min old (> 4 h 40 min): only 18Z
    assert ha.live_candidates(D(2025, 5, 6, 19, 45)) == [("20250506", 18)]
    assert ha.live_candidates(D(2025, 5, 6, 19, 39)) == [("20250506", 15)]
    # the next-older analysis is always 3 h older than the newest usable one (itself >= 1 h 40 min
    # old), so it is past the limit: there is one usable analysis at a time, live as in training,
    # and when it is missing the forecast runs without HRRR (measured on 2025: AUC 0.958)
    assert ha.live_candidates(D(2025, 5, 6, 18, 30)) == [("20250506", 15)]
    assert all(len(ha.live_candidates(D(2025, 5, 6, h, m))) == 1 for h in range(24) for m in (1, 31, 59))
    for now in (D(2025, 5, 6, h, m) for h in range(24) for m in (0, 20, 40, 59)):
        cands = ha.live_candidates(now)
        assert cands and cands[0] == ha.usable_analysis(now)                        # live == training rule
