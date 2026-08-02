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


def test_chains_across_a_cluster_within_gap_of_neighbour():
    # each consecutive pair is within min_gap, even though first and last are not
    result = merge_nearby_candidates([100, 108, 116], min_gap=10)
    assert result == [108]


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
