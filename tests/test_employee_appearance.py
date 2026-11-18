"""Same-day body-appearance gallery: threshold, margin, top-3 mean, daily rollover.

Run: .venv/Scripts/python.exe tests/test_employee_appearance.py
"""
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.ai.employee_appearance import MAX_SAMPLES, EmployeeAppearanceGallery

TESTS = []
NOON = datetime(2026, 3, 10, 7, 0)  # 12:00 in the default +05:00 workplace offset


def test(fn):
    TESTS.append(fn)
    return fn


def vector(*components):
    result = np.zeros(8, dtype=np.float32)
    result[:len(components)] = components
    return result


def with_similarity(base, similarity):
    """A unit vector whose cosine with `base` (a unit basis vector 0) is `similarity`."""
    other = np.zeros(8, dtype=np.float32)
    other[0], other[1] = similarity, float(np.sqrt(1 - similarity ** 2))
    return other


@test
def clear_match_returns_the_employee():
    gallery = EmployeeAppearanceGallery()
    gallery.learn(1, vector(1), NOON)
    gallery.learn(2, vector(0, 1), NOON)
    assert gallery.match(with_similarity(vector(1), .95), NOON) == 1


@test
def below_threshold_is_unknown():
    gallery = EmployeeAppearanceGallery()
    gallery.learn(1, vector(1), NOON)
    assert gallery.match(with_similarity(vector(1), .79), NOON) is None
    assert gallery.match(with_similarity(vector(1), .81), NOON) == 1


@test
def two_close_employees_are_ambiguous():
    gallery = EmployeeAppearanceGallery()
    gallery.learn(1, vector(1), NOON)
    gallery.learn(2, with_similarity(vector(1), .98), NOON)
    assert gallery.match(vector(1), NOON) is None  # 1.0 vs .98: lead below the 0.05 margin
    gallery.learn(3, vector(0, 0, 1), NOON)
    assert gallery.match(vector(0, 0, 1), NOON) == 3


@test
def score_is_the_mean_of_the_three_best_samples():
    gallery = EmployeeAppearanceGallery()
    for similarity in (1., .9, .8, .1, .1, .1):
        gallery.learn(1, with_similarity(vector(1), similarity), NOON)
    # top 3 = 1.0, .9, .8 -> mean .9; the weak samples do not drag the score down.
    assert gallery.match(vector(1), NOON) == 1
    thin = EmployeeAppearanceGallery()
    thin.learn(1, with_similarity(vector(1), .5), NOON)  # one sample: mean of what exists
    assert thin.match(vector(1), NOON) is None
    with patch("app.ai.employee_appearance.settings.reid_employee_threshold", .4):
        assert thin.match(vector(1), NOON) == 1


@test
def sample_count_per_employee_is_bounded_and_zero_vectors_ignored():
    gallery = EmployeeAppearanceGallery()
    for _ in range(MAX_SAMPLES + 20):
        gallery.learn(1, vector(1), NOON)
    gallery.learn(2, np.zeros(8, dtype=np.float32), NOON)
    assert len(gallery._samples[1]) == MAX_SAMPLES and 2 not in gallery._samples
    assert gallery.match(np.zeros(8, dtype=np.float32), NOON) is None


@test
def local_date_rollover_clears_the_gallery():
    gallery = EmployeeAppearanceGallery()
    gallery.learn(1, vector(1), NOON)
    assert gallery.match(vector(1), NOON + timedelta(hours=10)) == 1  # 22:00 local, same day
    # 19:00 UTC is 00:00 the next day at +05:00.
    assert gallery.match(vector(1), datetime(2026, 3, 10, 19, 0)) is None
    gallery.learn(1, vector(1), datetime(2026, 3, 10, 19, 0))
    assert gallery.match(vector(1), datetime(2026, 3, 10, 19, 5)) == 1


@test
def aware_timestamps_are_normalized_to_utc():
    gallery = EmployeeAppearanceGallery()
    gallery.learn(1, vector(1), NOON.replace(tzinfo=timezone.utc))
    assert gallery.match(vector(1), NOON) == 1


if __name__ == "__main__":
    failures = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(bool(failures))
