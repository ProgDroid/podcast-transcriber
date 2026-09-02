"""Assigning a cluster to an enrolled voice, or to nobody.

`None` is a RESULT, not a failure. Naming a cluster that is not one of the
enrolled people is the precision violation that matters most, so every
refusal path here is tested as carefully as every match.
"""

from __future__ import annotations

import math

from corpus.enrolment import (
    assign,
    build_centroids,
    co_membership_conflicts,
    score,
)

A = [1.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0]
NEAR_A = [math.sqrt(0.99), math.sqrt(0.01), 0.0]

LABELS = {
    "clip-a1": {"person": "Jacob Shapiro", "basis": "voice"},
    "clip-a2": {"person": "Jacob Shapiro", "basis": "voice"},
    "clip-b1": {"person": "Rob Larity", "basis": "voice"},
    "clip-ctx": {"person": "Marko Papic", "basis": "context"},
    "clip-skip": {"person": None, "basis": "skipped"},
}
VECTORS = {
    "clip-a1": A,
    "clip-a2": A,
    "clip-b1": B,
    "clip-ctx": B,
    "clip-skip": A,
}


def test_builds_one_centroid_per_named_person():
    centroids = build_centroids(LABELS, VECTORS)
    assert set(centroids) == {"Jacob Shapiro", "Rob Larity"}


def test_excludes_context_labels_from_enrolment():
    # speaker_tool.py:705 warns that a label inferred from episode text scores
    # a matcher correct for the wrong reason. Part 8.4 forecloses every
    # text-only route to speaker identity; importing one here would smuggle it
    # back in as voice ground truth.
    assert "Marko Papic" not in build_centroids(LABELS, VECTORS)


def test_excludes_skipped_labels_from_enrolment():
    centroids = build_centroids(LABELS, VECTORS)
    assert None not in centroids
    assert len(centroids) == 2


def test_ignores_labels_with_no_vector():
    centroids = build_centroids(LABELS, {"clip-a1": A})
    assert set(centroids) == {"Jacob Shapiro"}


def test_centroids_are_unit_length():
    for vector in build_centroids(LABELS, VECTORS).values():
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_score_orders_candidates_best_first():
    ranked = score(NEAR_A, {"Jacob Shapiro": A, "Rob Larity": B})
    assert [name for name, _ in ranked] == ["Jacob Shapiro", "Rob Larity"]


def test_assign_returns_the_best_match_when_clear():
    centroids = {"Jacob Shapiro": A, "Rob Larity": B}
    assert assign(NEAR_A, centroids, threshold=0.5, margin=0.1) == "Jacob Shapiro"


def test_assign_refuses_below_the_threshold():
    centroids = {"Jacob Shapiro": A, "Rob Larity": B}
    assert assign(NEAR_A, centroids, threshold=0.999, margin=0.0) is None


def test_assign_refuses_when_two_candidates_are_a_hair_apart():
    # The dangerous case: a threshold alone resolves this confidently and
    # wrongly. Both score well; neither wins by enough.
    tie = [math.sqrt(0.5), math.sqrt(0.5), 0.0]
    centroids = {"Jacob Shapiro": A, "Rob Larity": B}
    assert assign(tie, centroids, threshold=0.5, margin=0.05) is None


def test_assign_returns_none_when_there_are_no_centroids():
    assert assign(A, {}, threshold=0.5, margin=0.1) is None


def test_assign_matches_a_lone_candidate_that_clears_the_threshold():
    # With one enrolled voice there is no runner-up, so margin cannot apply.
    assert assign(A, {"Jacob Shapiro": A}, threshold=0.9, margin=0.5) == "Jacob Shapiro"


def test_co_membership_flags_two_clusters_of_one_episode_sharing_a_person():
    assignments = [
        {"episode": "ep1.txt", "cluster": "SPEAKER_00", "person": "Jacob Shapiro"},
        {"episode": "ep1.txt", "cluster": "SPEAKER_01", "person": "Jacob Shapiro"},
        {"episode": "ep2.txt", "cluster": "SPEAKER_00", "person": "Jacob Shapiro"},
    ]
    conflicts = co_membership_conflicts(assignments)
    assert len(conflicts) == 1
    assert conflicts[0]["episode"] == "ep1.txt"
    assert conflicts[0]["person"] == "Jacob Shapiro"
    assert sorted(conflicts[0]["clusters"]) == ["SPEAKER_00", "SPEAKER_01"]


def test_co_membership_ignores_unassigned_clusters():
    # Two refusals in one episode are not a conflict -- they are two
    # independent "we do not know", which is the designed behaviour.
    assignments = [
        {"episode": "ep1.txt", "cluster": "SPEAKER_00", "person": None},
        {"episode": "ep1.txt", "cluster": "SPEAKER_01", "person": None},
    ]
    assert co_membership_conflicts(assignments) == []


def test_co_membership_is_empty_when_each_person_appears_once_per_episode():
    assignments = [
        {"episode": "ep1.txt", "cluster": "SPEAKER_00", "person": "Jacob Shapiro"},
        {"episode": "ep1.txt", "cluster": "SPEAKER_01", "person": "Rob Larity"},
    ]
    assert co_membership_conflicts(assignments) == []
