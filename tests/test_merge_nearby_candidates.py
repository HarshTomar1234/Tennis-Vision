"""
tests/test_merge_nearby_candidates.py
Unit tests for merge_nearby_candidates -- collapses duplicate detections of the same
real event from the union of two independent candidate generators (docs/journal/0018).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hit_bounce_classifier import merge_nearby_candidates


def test_merges_a_tight_cluster_into_one_frame():
    result = merge_nearby_candidates([100, 103, 106], min_gap=10)
    assert result == [103]


def test_keeps_far_apart_candidates_separate():
    result = merge_nearby_candidates([50, 200, 400], min_gap=10)
    assert result == [50, 200, 400]


def test_does_not_chain_beyond_min_gap():
    """
    A cluster must not grow wider than min_gap, even when consecutive members are close.

    This test previously asserted the opposite, that 100, 108 and 116 collapse to a single
    event at 108, on the reasoning that each consecutive pair is within min_gap. Chaining
    was deliberate and it was wrong: 100 and 116 are 16 frames apart, which at 30fps is
    0.53s, comfortably long enough to hold two separate contacts. In a dense rally the
    chain runs much further than three candidates and deletes real events.

    Measured across 12 dataset clips and 91 labelled contacts, bounding the cluster moved
    event recall from 51.6% to 68.1% with precision unchanged at ~94%, because everything
    it stopped destroying was real. See eval/event_recall_funnel.py.
    """
    result = merge_nearby_candidates([100, 108, 116], min_gap=10)
    assert result == [108, 116], "cluster chained past min_gap"


def test_cluster_spans_at_most_min_gap():
    """The invariant the bound exists to guarantee, stated directly."""
    candidates = list(range(0, 60, 4))       # 0, 4, 8, ... 56: every step is under min_gap
    result = merge_nearby_candidates(candidates, min_gap=10)
    # Chaining would return a single representative for all 15 candidates.
    assert len(result) > 1, "chained the whole sequence into one event"
    assert len(result) >= len(candidates) // 4


def test_even_sized_cluster_picks_a_real_member():
    result = merge_nearby_candidates([100, 105], min_gap=10)
    assert result[0] in (100, 105)
    assert len(result) == 1


def test_empty_input_returns_empty():
    assert merge_nearby_candidates([]) == []


def test_unordered_input_still_clusters_correctly():
    result = merge_nearby_candidates([106, 100, 103], min_gap=10)
    assert result == [103]


def test_duplicate_frames_collapse_too():
    result = merge_nearby_candidates([100, 100, 100], min_gap=10)
    assert result == [100]
