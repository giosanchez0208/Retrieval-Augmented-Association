"""Offline post-processing of finished tracks."""

from __future__ import annotations

import numpy as np

from reidtrack.data.mot import Tracks


def interpolate(tracks: Tracks, max_gap: int = 20) -> Tracks:
    """Fill gaps of up to ``max_gap`` missing frames inside each track with straight-line
    boxes between the sightings on either side.

    This needs frames after the gap, so it runs after tracking, not in real time.
    Filled boxes take the lower score of the two sightings.
    """
    if len(tracks) == 0 or max_gap <= 0:
        return tracks
    frames, ids, boxes, scores = [tracks.frame], [tracks.track_id], [tracks.xyxy], [tracks.score]
    order = np.lexsort((tracks.frame, tracks.track_id))
    f, t, b, s = tracks.frame[order], tracks.track_id[order], tracks.xyxy[order], tracks.score[order]
    gap = np.diff(f) - 1
    fill = np.flatnonzero((t[1:] == t[:-1]) & (gap > 0) & (gap <= max_gap))
    for i in fill:
        steps = np.arange(1, gap[i] + 1)
        w = (steps / (gap[i] + 1))[:, None]
        frames.append(f[i] + steps)
        ids.append(np.full(len(steps), t[i]))
        boxes.append((1 - w) * b[i] + w * b[i + 1])
        scores.append(np.full(len(steps), min(s[i], s[i + 1])))
    out = Tracks(
        frame=np.concatenate(frames).astype(np.int32),
        track_id=np.concatenate(ids).astype(np.int32),
        xyxy=np.concatenate(boxes).astype(np.float32),
        score=np.concatenate(scores).astype(np.float32),
    )
    return out.select(np.lexsort((out.track_id, out.frame)))
