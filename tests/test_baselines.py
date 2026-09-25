import numpy as np

from reidtrack.baselines import ByteTrack, Sort


def walk(start, step, frames):
    """Boxes of one person walking at constant velocity."""
    return [np.array(start, dtype=float) + i * np.array(step, dtype=float) for i in range(frames)]


def run(tracker, frames):
    """frames: list of (boxes, scores) per frame. Returns per-frame id lists."""
    out = []
    for boxes, scores in frames:
        ids, _, _ = tracker.update(np.array(boxes, dtype=float).reshape(-1, 4), np.array(scores, dtype=float))
        out.append(ids.tolist())
    return out


def test_sort_keeps_one_id_for_a_walking_person():
    boxes = walk([100, 100, 150, 250], [4, 0, 4, 0], 10)
    ids = run(Sort(), [([b], [0.9]) for b in boxes])

    assert all(frame == [1] for frame in ids)


def test_sort_needs_min_hits_after_the_first_frames():
    first = walk([100, 100, 150, 250], [4, 0, 4, 0], 6)
    late = walk([400, 100, 450, 250], [4, 0, 4, 0], 4)
    frames = [([b], [0.9]) for b in first[:4]] + [([first[4 + i], late[i]], [0.9, 0.9]) for i in range(2)]
    frames += [([late[2 + i]], [0.9]) for i in range(2)]

    ids = run(Sort(min_hits=3), frames)

    # created at frame 4 with a streak of 0; matched at 5, 6, 7
    assert ids[4] == [1] and ids[5] == [1] and ids[6] == []
    assert ids[7] == [2]


def test_sort_forgets_after_one_missed_frame():
    boxes = walk([100, 100, 150, 250], [4, 0, 4, 0], 8)
    frames = [([b], [0.9]) for b in boxes[:4]] + [([], [])] * 2 + [([b], [0.9]) for b in boxes[6:]]

    ids = run(Sort(min_hits=1), frames)

    # track 1 dies after two missed frames; the returning person starts over as 2
    assert ids[3] == [1] and ids[6] == [] and ids[7] == [2]


def test_bytetrack_keeps_a_person_through_a_low_score_detection():
    boxes = walk([100, 100, 150, 250], [4, 0, 4, 0], 8)
    scores = [0.9, 0.9, 0.9, 0.3, 0.3, 0.9, 0.9, 0.9]  # partly occluded at frames 4-5

    ids = run(ByteTrack(min_box_area=0), [([b], [s]) for b, s in zip(boxes, scores)])

    assert ids[0] == [1] and ids[1] == [1]
    assert all(frame == [1] for frame in ids[3:])


def test_bytetrack_recovers_a_lost_track_within_the_buffer():
    boxes = walk([100, 100, 150, 250], [2, 0, 2, 0], 20)
    frames = [([b], [0.9]) for b in boxes[:5]] + [([], [])] * 10 + [([b], [0.9]) for b in boxes[15:]]

    ids = run(ByteTrack(min_box_area=0), frames)

    assert ids[4] == [1] and ids[5:15] == [[]] * 10
    assert ids[15] == [1]  # same id after 10 missing frames


def test_bytetrack_forgets_after_the_buffer():
    boxes = walk([100, 100, 150, 250], [0, 0, 0, 0], 60)
    frames = [([b], [0.9]) for b in boxes[:5]] + [([], [])] * 40 + [([b], [0.9]) for b in boxes[45:]]

    ids = run(ByteTrack(min_box_area=0, track_buffer=30), frames)

    assert ids[4] == [1]
    assert ids[46] == [2]  # a fresh track, confirmed on its second frame


def test_bytetrack_filters_wide_and_tiny_output_boxes():
    ids = run(ByteTrack(), [([[0, 0, 100, 40]], [0.9]), ([[0, 0, 5, 10]], [0.9])])

    assert ids == [[], []]
