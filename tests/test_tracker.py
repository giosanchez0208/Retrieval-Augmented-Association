import numpy as np

from reidtrack.memory import AppearanceMemory, Entry, Regime, exiting, similarity
from reidtrack.tracker import RetrievalTracker, TrackerConfig


def unit(i, dim=16, mix=None):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1
    if mix is not None:
        v[mix] = 0.3
    return v / np.linalg.norm(v)


def entry():
    return Entry(1, np.zeros(8), np.eye(8), first_seen=0, last_seen=0, score=0.9)


def test_memory_keeps_distinct_views_and_skips_near_duplicates():
    mem, e = AppearanceMemory(max_prototypes=2, min_novelty=0.1), entry()
    mem.write(e, unit(0))
    mem.write(e, unit(0, mix=1))  # cosine ~0.96 to the first: not new
    assert len(e.prototypes) == 1
    mem.write(e, unit(2))
    mem.write(e, unit(3))  # full: replaces a view only if it is more distinct than the most redundant one
    assert len(e.prototypes) == 2
    assert similarity([e], np.stack([unit(2)]))[0, 0] > 0.99 or similarity([e], np.stack([unit(3)]))[0, 0] > 0.99


def test_similarity_is_the_best_view():
    e = entry()
    e.prototypes = [unit(0), unit(1)]
    np.testing.assert_allclose(similarity([e], np.stack([unit(1), unit(5)])), [[1, 0]], atol=1e-6)


def test_exiting_needs_the_border_and_outward_motion():
    assert exiting(np.array([0, 100, 50, 300]), np.array([-2, 0]), 1920, 1080)
    assert not exiting(np.array([0, 100, 50, 300]), np.array([2, 0]), 1920, 1080)
    assert not exiting(np.array([900, 100, 950, 300]), np.array([-2, 0]), 1920, 1080)


def box_at(x, y=300, w=60, h=160):
    return [x, y, x + w, y + h]


def run(tracker, frames):
    out = []
    for boxes, feats, scores in frames:
        b = np.array(boxes, dtype=float).reshape(-1, 4)
        f = np.array(feats, dtype=np.float32).reshape(len(b), -1) if len(b) else np.zeros((0, 16), np.float32)
        s = np.array(scores, dtype=float) if scores is not None else np.full(len(b), 0.9)
        ids, _, _ = tracker.update(b, s, f)
        out.append(ids.tolist())
    return out


def walker(start, step, n, feat, score=0.9):
    return [([box_at(start + i * step)], [feat], [score]) for i in range(n)]


def test_one_id_for_a_walking_person():
    ids = run(RetrievalTracker(1920, 1080), walker(400, 5, 30, unit(0)))
    assert all(f == [1] for f in ids)


def test_a_hidden_person_is_recalled_by_appearance():
    frames = walker(400, 5, 20, unit(0))
    frames += [([], [], [])] * 30  # 1 s hidden
    frames += [([box_at(400 + 50 * 5 + 40)], [unit(0)], [0.9])] * 3  # reappears a little off the prediction
    ids = run(RetrievalTracker(1920, 1080), frames)
    assert ids[-1] == [1]


def test_a_stranger_in_the_predicted_spot_gets_a_new_id():
    frames = walker(400, 0, 20, unit(0))
    frames += [([], [], [])] * 10
    frames += [([box_at(400)], [unit(7)], [0.9])] * 3
    ids = run(RetrievalTracker(1920, 1080), frames)
    assert ids[-1] == [2]


def test_exits_are_final_in_benchmark_mode_and_recalled_in_reentry_mode():
    out = walker(1700, 10, 20, unit(0))  # walks out through the right border
    back = [([box_at(1840, w=60)], [unit(0)], [0.9])] * 3
    frames = out + [([], [], [])] * 90 + back  # 3 s away, returns at the same edge

    benchmark = run(RetrievalTracker(1920, 1080), frames)
    reentry = run(RetrievalTracker(1920, 1080, config=TrackerConfig(reentry=True)), frames)

    assert benchmark[-1] == [2]
    assert reentry[-1] == [1]


def test_low_score_detections_keep_an_active_person():
    frames = walker(400, 5, 10, unit(0)) + walker(450, 5, 5, unit(0), score=0.3) + walker(475, 5, 5, unit(0))
    ids = run(RetrievalTracker(1920, 1080), frames)
    assert all(f == [1] for f in ids[:10]) and ids[-1] == [1]


def test_crowded_sightings_do_not_write_appearance():
    tracker = RetrievalTracker(1920, 1080)
    run(tracker, walker(400, 0, 5, unit(0)))
    before = [p.copy() for p in tracker.entries[0].prototypes]
    # a second person overlapping heavily: the first person's new view must not be stored
    run(tracker, [([box_at(400), box_at(410)], [unit(5), unit(6)], [0.9, 0.9])])
    assert len(tracker.entries[0].prototypes) == len(before)
    assert tracker.entries[0].regime == Regime.ACTIVE


def test_hidden_people_can_be_reported_for_a_while():
    frames = walker(400, 5, 20, unit(0)) + [([], [], [])] * 40
    quiet = run(RetrievalTracker(1920, 1080), frames)
    shown = run(RetrievalTracker(1920, 1080, config=TrackerConfig(emit_hidden=0.5)), frames)

    assert quiet[20:] == [[]] * 40
    assert shown[20:35] == [[1]] * 15  # predicted boxes for half a second at 30 fps
    assert shown[36:] == [[]] * 24


def test_interpolation_fills_short_gaps_only():
    from reidtrack.data.mot import Tracks
    from reidtrack.track.postprocess import interpolate

    tracks = Tracks(
        frame=np.array([1, 4, 30], np.int32), track_id=np.array([7, 7, 7], np.int32),
        xyxy=np.array([[0, 0, 10, 10], [30, 0, 40, 10], [30, 0, 40, 10]], np.float32), score=np.array([0.9, 0.6, 0.8], np.float32),
    )
    out = interpolate(tracks, max_gap=5)

    assert out.frame.tolist() == [1, 2, 3, 4, 30]
    np.testing.assert_allclose(out.xyxy[1], [10, 0, 20, 10])
    assert out.score[1] == np.float32(0.6)


def test_hysteresis_keeps_a_pairing_until_a_rival_wins_clearly():
    def slightly_prefers_the_other_box(cues):  # the box where the person was scores 0.60, a far one 0.62
        return np.tile([0.60, 0.62], (cues.shape[0], 1))

    frames = [([box_at(400)], [unit(0)], [0.9]), ([box_at(400), box_at(1200)], [unit(0), unit(0)], [0.9, 0.9])]
    plain = RetrievalTracker(1920, 1080, reranker=slightly_prefers_the_other_box)
    sticky = RetrievalTracker(1920, 1080, config=TrackerConfig(hysteresis=0.1), reranker=slightly_prefers_the_other_box)
    run(plain, frames)
    run(sticky, frames)

    assert plain.assignments[1] == 1  # without it, the person jumps to the slightly better box
    assert sticky.assignments[1] == 0
