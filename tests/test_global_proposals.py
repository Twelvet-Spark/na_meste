"""Pure appearance matching; durable journal/worker integration is separate.

Run: .venv/Scripts/python.exe tests/test_global_proposals.py
"""
import sys
import traceback
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid5, NAMESPACE_URL

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.ai.global_tracker import CameraTopology, EmbeddingItem, GlobalIdentityManager
from app.config import settings

TESTS = []
STAMP = datetime(2026, 11, 16, 10)


def test(fn):
    TESTS.append(fn)
    return fn


def vector(index=0, cosine=None):
    result = np.zeros(512, dtype=np.float32)
    if cosine is None:
        result[index] = 1.
    else:
        result[0] = cosine
        result[index] = np.sqrt(1. - cosine * cosine)
    return result


def item(number=1, *, camera_id=1, generation=1, now=10., **changes):
    value = EmbeddingItem(
        visit_id=str(uuid5(NAMESPACE_URL, f"test-visit/{camera_id}/{generation}/{number}")),
        camera_id=camera_id,
        tracking_generation=str(uuid5(NAMESPACE_URL, f"test-generation/{camera_id}/{generation}")),
        track_id=number, source_revision=1, stream_id=f"test-stream-{camera_id}-{generation}",
        packet_seq=1, observed_at=STAMP + timedelta(seconds=now), received_mono=now,
        bbox=(.1, .1, .4, .9), aspect=2.5, crop_quality=.9,
        embedding=vector(), employee_id=None,
    )
    return replace(value, **changes)


def ack(proposal, global_id=731, **changes):
    values = dict(
        kind="LINK_COMMITTED", camera_id=proposal.camera_id,
        tracking_generation=proposal.tracking_generation, visit_id=proposal.visit_id,
        track_id=proposal.track_id, source_revision=proposal.source_revision,
        stream_id=proposal.stream_id, packet_seq=proposal.packet_seq,
        global_id=global_id, link_key=proposal.link_key, reason=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def commit(manager, value, global_id=731):
    proposals = manager.process_embeddings([value])
    assert len(proposals) == 1
    manager.acknowledge(ack(proposals[0], global_id))
    return proposals[0]


@test
def test_proposals_are_not_bindings_or_matching_gallery_until_commit():
    manager = GlobalIdentityManager()
    first, second = item(), item(camera_id=2, now=15.)
    proposals = manager.process_embeddings([first, second])
    assert len(proposals) == 2 and all(p.global_id is None for p in proposals)
    assert manager.global_id_for(first.visit_id) is None
    assert manager.camera_tracks(1) == [] and manager.snapshot() == []
    manager.acknowledge(ack(proposals[0], 731))
    assert manager.global_id_for(first.visit_id) == 731
    assert manager.global_id_for(second.visit_id) is None
    assert manager.camera_tracks(1)[0]["global_id"] == 731
    third = manager.process_embeddings([item(camera_id=3, now=16.)])[0]
    assert third.global_id == 731, "Only the acknowledged vector is matchable"


@test
def test_retry_keeps_stable_receipt_and_ack_publishes_only_once():
    manager = GlobalIdentityManager()
    value = item()
    first = manager.process_embeddings([value])[0]
    retry = manager.process_embeddings([value])[0]
    assert retry is first and str(UUID(first.link_key)) == first.link_key
    manager.acknowledge(ack(first, 9127))
    before = [(row["global_id"], row["embeddings"]) for row in manager.snapshot()]
    manager.acknowledge(ack(first, 9127))
    manager.acknowledge(ack(first, 9999))
    assert manager.global_id_for(value.visit_id) == 9127
    assert [(row["global_id"], row["embeddings"]) for row in manager.snapshot()] == before == [(9127, 1)]
    assert manager.process_embeddings([value]) == [], "An acknowledged packet is not new evidence"
    independent = GlobalIdentityManager().process_embeddings([value])[0]
    assert independent.link_key == first.link_key
    newer = manager.process_embeddings([replace(value, packet_seq=2, received_mono=11.)])[0]
    assert newer.link_key != first.link_key and newer.global_id == 9127


@test
def test_wrong_ack_kind_or_any_wrong_token_never_publishes():
    for changes in (
        {"kind": "COMMITTED"}, {"visit_id": item(2).visit_id}, {"camera_id": 2},
        {"tracking_generation": item(generation=2).tracking_generation}, {"track_id": 2},
        {"source_revision": 2}, {"stream_id": "retired-stream"}, {"packet_seq": 2},
        {"link_key": str(uuid5(NAMESPACE_URL, "wrong-receipt"))},
        {"global_id": None}, {"global_id": 0}, {"global_id": True},
    ):
        manager = GlobalIdentityManager()
        value = item()
        proposal = manager.process_embeddings([value])[0]
        manager.acknowledge(ack(proposal, **changes))
        assert manager.global_id_for(value.visit_id) is None, changes
        assert manager.snapshot() == [], changes
        manager.acknowledge(ack(proposal, 87))
        assert manager.global_id_for(value.visit_id) == 87


@test
def test_superseded_packet_and_dropped_generation_acks_are_rejected():
    manager = GlobalIdentityManager()
    value = item()
    old = manager.process_embeddings([value])[0]
    current = manager.process_embeddings([replace(value, packet_seq=2, received_mono=11.)])[0]
    manager.acknowledge(ack(old))
    assert manager.global_id_for(value.visit_id) is None
    manager.acknowledge(ack(current, 42))
    assert manager.global_id_for(value.visit_id) == 42
    pending = manager.process_embeddings([item(2, now=12.)])[0]
    manager.drop_camera(1)
    next_generation = item(generation=2, now=13., embedding=vector(1))
    next_proposal = manager.process_embeddings([next_generation])[0]
    manager.acknowledge(ack(pending, 43))
    assert manager.global_id_for(pending.visit_id) is None
    assert manager.global_id_for(next_generation.visit_id) is None
    manager.acknowledge(ack(next_proposal, 44))
    assert manager.global_id_for(next_generation.visit_id) == 44


@test
def test_older_observation_cannot_replace_retained_newer_packet():
    manager = GlobalIdentityManager()
    value = item(packet_seq=2, now=11.)
    current = manager.process_embeddings([value])[0]
    assert manager.process_embeddings([replace(value, packet_seq=1, received_mono=10.)]) == []
    manager.acknowledge(ack(current, 82))
    assert manager.global_id_for(value.visit_id) == 82


@test
def test_pending_and_gallery_vectors_do_not_alias_mutable_caller_memory():
    manager = GlobalIdentityManager()
    value = item()
    proposal = manager.process_embeddings([value])[0]
    value.embedding[:] = vector(1)
    assert proposal.embedding[0] == 1. and not proposal.embedding.flags.writeable
    try:
        proposal.global_id = 123
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("Proposal metadata must be frozen")
    manager.acknowledge(ack(proposal))
    candidate = manager.process_embeddings([item(camera_id=2, now=15.)])[0]
    assert candidate.global_id == 731


@test
def test_body_never_supplies_employee_identity_and_face_conflict_vetoes_match():
    manager = GlobalIdentityManager()
    commit(manager, item(employee_id=7))
    unknown = item(camera_id=2, now=15.)
    proposal = manager.process_embeddings([unknown])[0]
    assert proposal.global_id == 731 and proposal.employee_id is None
    manager.acknowledge(ack(proposal))
    assert unknown.employee_id is None
    assert manager.snapshot()[0]["employee_id"] == 7, "Only confirmed face context is retained"
    conflict = manager.process_embeddings([item(camera_id=3, now=20., employee_id=8)])[0]
    assert conflict.global_id is None and conflict.employee_id == 8
    bound_conflict = replace(unknown, packet_seq=2, received_mono=21., employee_id=8)
    assert manager.process_embeddings([bound_conflict]) == [], "A linked visit cannot silently relink"
    assert manager.global_id_for(unknown.visit_id) == 731


@test
def test_same_camera_occupancy_blocks_merge_until_real_overlap_window_expires():
    manager = GlobalIdentityManager()
    commit(manager, item())
    overlap = manager.process_embeddings([item(2, now=11.)])[0]
    assert overlap.global_id is None
    after = manager.process_embeddings([item(3, now=12.1)])[0]
    assert after.global_id == 731


@test
def test_same_batch_pending_candidates_do_not_poison_same_camera_gallery():
    manager = GlobalIdentityManager()
    proposals = manager.process_embeddings([item(), item(2)])
    assert len(proposals) == 2 and [p.global_id for p in proposals] == [None, None]
    manager.acknowledge(ack(proposals[0], 41))
    manager.acknowledge(ack(proposals[1], 42))
    assert {p["global_id"] for p in manager.camera_tracks(1)} == {41, 42}


@test
def test_topology_transition_window_is_a_veto_not_just_a_score():
    topology = CameraTopology({(1, 2): (4., 10.)})
    manager = GlobalIdentityManager(topology)
    commit(manager, item())
    assert manager.process_embeddings([item(camera_id=2, now=12.)])[0].global_id is None
    assert manager.process_embeddings([item(camera_id=2, number=2, now=15.)])[0].global_id == 731
    assert manager.process_embeddings([item(camera_id=2, number=3, now=21.)])[0].global_id is None


@test
def test_cosine_and_composite_thresholds_both_gate_matches():
    manager = GlobalIdentityManager()
    commit(manager, item())
    with patch.object(settings, "reid_similarity_threshold", .8):
        assert manager.process_embeddings([item(camera_id=2, now=15., embedding=vector(1, .79))])[0].global_id is None
        with patch.object(settings, "global_match_threshold", .99):
            assert manager.process_embeddings([item(2, camera_id=2, now=15.)])[0].global_id is None
        assert manager.process_embeddings([item(3, camera_id=2, now=15.)])[0].global_id == 731


@test
def test_aspect_changes_composite_without_becoming_an_exclusion_mask():
    manager = GlobalIdentityManager()
    commit(manager, item(aspect=2.))
    with patch.object(settings, "w_reid", .5), patch.object(settings, "w_temporal", 0.), \
            patch.object(settings, "w_topology", 0.), patch.object(settings, "w_aspect", .5), \
            patch.object(settings, "w_face", 0.), patch.object(settings, "global_match_threshold", .9):
        assert manager.process_embeddings([item(camera_id=2, now=15., aspect=2.)])[0].global_id == 731
        seated = manager.process_embeddings([item(2, camera_id=3, now=15., aspect=.5)])[0]
        assert seated.global_id is None, "A dissimilar aspect may create a separate identity, never erase a visit"


@test
def test_close_second_candidate_remains_tentative_instead_of_merging():
    manager = GlobalIdentityManager()
    proposals = manager.process_embeddings([item(), item(2)])
    manager.acknowledge(ack(proposals[0], 71))
    manager.acknowledge(ack(proposals[1], 72))
    ambiguous = manager.process_embeddings([item(camera_id=2, now=15.)])[0]
    assert ambiguous.global_id is None
    assert {p["global_id"] for p in manager.snapshot()} == {71, 72}


@test
def test_face_agreement_is_a_bonus_not_a_body_identity_assignment():
    manager = GlobalIdentityManager()
    commit(manager, item(employee_id=7))
    unknown = manager.process_embeddings([item(camera_id=2, now=15.)])[0]
    known = manager.process_embeddings([item(2, camera_id=3, now=15., employee_id=7)])[0]
    assert known.global_id == unknown.global_id == 731
    assert known.score > unknown.score
    assert unknown.employee_id is None


@test
def test_ttl_expires_matching_gallery_bindings_and_unacknowledged_work():
    manager = GlobalIdentityManager()
    with patch.object(settings, "global_gallery_ttl", 20.), patch.object(settings, "track_lost_timeout", 3.):
        first = item()
        commit(manager, first)
        pending = manager.process_embeddings([item(2, now=11.)])[0]
        manager.sweep(20.)
        assert manager.global_id_for(first.visit_id) is None
        manager.acknowledge(ack(pending, 991))
        assert manager.global_id_for(pending.visit_id) is None
        assert manager.process_embeddings([item(camera_id=2, now=21.)])[0].global_id == 731
        manager.sweep(31.)
        assert manager.snapshot() == []
        assert manager.process_embeddings([item(camera_id=3, now=32.)])[0].global_id is None


@test
def test_history_is_bounded_and_only_committed_refreshes_add_vectors():
    manager = GlobalIdentityManager()
    with patch.object(settings, "global_history_len", 2):
        value = item()
        commit(manager, value)
        for sequence in (2, 3, 4):
            proposal = manager.process_embeddings([replace(value, packet_seq=sequence, received_mono=10. + sequence)])[0]
            assert manager.snapshot()[0]["embeddings"] <= 2
            manager.acknowledge(ack(proposal))
        assert manager.snapshot()[0]["embeddings"] == 2


@test
def test_capacity_rejects_new_work_without_evicting_retained_acknowledgements():
    manager = GlobalIdentityManager()
    values = [item(number=number) for number in range(1, 4097)]
    proposals = manager.process_embeddings(values)
    assert len(proposals) == 4096
    assert manager.process_embeddings([item(4097)]) == []
    assert manager.process_embeddings([values[0]])[0] is proposals[0]
    manager.acknowledge(ack(proposals[0], 99))
    assert manager.global_id_for(values[0].visit_id) == 99
    assert manager.process_embeddings([item(4097)]) == [], "Pending and bindings share the live-visit bound"
    manager.drop_camera(1)
    assert len(manager.process_embeddings([item(4097, generation=2)])) == 1
    manager.acknowledge(ack(proposals[1], 100))
    assert manager.global_id_for(values[1].visit_id) is None


@test
def test_drop_camera_keeps_other_camera_binding_and_acknowledged_appearance():
    manager = GlobalIdentityManager()
    first, other = item(), item(camera_id=2, now=15.)
    commit(manager, first)
    commit(manager, other)
    pending = manager.process_embeddings([item(2, camera_id=2, now=16., embedding=vector(1))])[0]
    manager.drop_camera(1)
    assert manager.global_id_for(first.visit_id) is None
    assert manager.global_id_for(other.visit_id) == 731
    manager.acknowledge(ack(pending, 732))
    assert manager.global_id_for(pending.visit_id) == 732
    assert manager.process_embeddings([item(camera_id=3, now=20.)])[0].global_id == 731


@test
def test_notifications_require_exact_durable_ack_and_duplicates_are_silent():
    class Notifications:
        def __init__(self):
            self.messages = []

        def send_text(self, message):
            self.messages.append(message)

    notifier = Notifications()
    with patch.object(settings, "telegram_notify_global_new", True), \
            patch.object(settings, "telegram_notify_global_transition", True):
        manager = GlobalIdentityManager(notifier=notifier)
        proposal = manager.process_embeddings([item()])[0]
        assert notifier.messages == []
        manager.acknowledge(ack(proposal, 888, packet_seq=999))
        assert notifier.messages == []
        manager.acknowledge(ack(proposal, 888))
        assert len(notifier.messages) == 1 and "888" in notifier.messages[0]
        manager.acknowledge(ack(proposal, 888))
        transition = manager.process_embeddings([item(camera_id=2, now=15.)])[0]
        assert len(notifier.messages) == 1
        manager.acknowledge(ack(transition, 888))
        manager.acknowledge(ack(transition, 888))
        assert len(notifier.messages) == 2 and "888" in notifier.messages[1]


@test
def test_malformed_items_fail_before_any_partial_batch_publication():
    for changes in (
        {"visit_id": "not-a-uuid"}, {"tracking_generation": "not-a-uuid"},
        {"camera_id": True}, {"camera_id": 0}, {"track_id": -1}, {"source_revision": 0},
        {"packet_seq": -1}, {"stream_id": ""}, {"employee_id": True},
        {"received_mono": float("nan")}, {"observed_at": "2026-11-16"},
        {"bbox": (0., 0., 0., 1.)}, {"bbox": (0., 0., 2., 1.)},
        {"bbox": (0., 0., float("nan"), 1.)}, {"aspect": 0.},
        {"crop_quality": float("inf")}, {"embedding": np.zeros(512, np.float32)},
        {"embedding": np.ones(511, np.float32)}, {"embedding": np.full(512, np.nan)},
    ):
        manager = GlobalIdentityManager()
        try:
            manager.process_embeddings([item(), item(2, **changes)])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid metadata: {changes}")
        assert manager.snapshot() == [] and manager.camera_tracks(1) == []
        assert manager.process_embeddings([item()])[0].global_id is None


@test
def test_pending_links_reserve_same_camera_occupancy_without_publishing():
    manager = GlobalIdentityManager()
    commit(manager, item())
    first, overlapping = manager.process_embeddings([
        item(camera_id=2, now=15.), item(2, camera_id=2, now=15.1)])
    assert first.global_id == 731 and overlapping.global_id is None
    assert manager.camera_tracks(2) == []
    assert manager.global_id_for(first.visit_id) is None
    manager.acknowledge(ack(first))
    assert manager.global_id_for(first.visit_id) == 731


@test
def test_identity_capacity_discards_oldest_live_cache_not_durable_history():
    manager = GlobalIdentityManager()
    proposals = manager.process_embeddings([item(), item(2), item(3)])
    with patch("app.ai.global_tracker.MAX_IDENTITIES", 2):
        for proposal, global_id in zip(proposals, (41, 42, 43)):
            manager.acknowledge(ack(proposal, global_id))
        assert {row["global_id"] for row in manager.snapshot()} == {42, 43}
        assert manager.global_id_for(proposals[0].visit_id) is None
        assert manager.global_id_for(proposals[2].visit_id) == 43


@test
def test_writer_replay_id_is_authoritative_over_unpublished_candidate():
    manager = GlobalIdentityManager()
    commit(manager, item(), 71)
    value = item(camera_id=2, now=15.)
    proposal = manager.process_embeddings([value])[0]
    assert proposal.global_id == 71
    # A lost earlier acknowledgement may already have linked this visit in DB.
    # The writer replays that durable ID, not the matcher's tentative candidate.
    manager.acknowledge(ack(proposal, 92))
    assert manager.global_id_for(value.visit_id) == 92
    assert {row["global_id"] for row in manager.snapshot()} == {71, 92}


if __name__ == "__main__":
    failed = 0
    for case in TESTS:
        try:
            case()
            print(f"PASS {case.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {case.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(bool(failed))
