# Spatial-Aware Multi-Person Re-Identification

Multi-person tracking with a learned, spatial-aware association model and an
identity bank that keeps people through occlusions and re-identifies them when
they come back into view. Benchmarked on MOT17.

**Status:** early development. Data preparation is in place; baselines and
models come next.

## Background

This project builds on *Transformer-Based Single Person Re-Identification with
Spatial-Aware Matching* by Caine Ivan R. Bautista, Mark Angelo L. Gallardo and
Gio Kiefer A. Sanchez, presented at the 8th Collaborative Online International
Learning (COIL) program. That version scored one pair of people at a time. This
one associates everyone in the frame at once and adds a memory for people who
are hidden or out of view.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data

Extract the MOT17 release so that `MOT17/train` and `MOT17/test` sit at the
repository root, then run:

```bash
uv run python -m reidtrack.data.prepare --raw MOT17 --out data/mot17
```

The release stores every sequence three times, once per public detector (DPM,
FRCNN, SDP), with identical images and ground truth. The prepare step:

- checks that the three copies really are identical (image contents, ground
  truth, sequence info) and stops without writing anything if they are not;
- writes one folder per sequence to `data/mot17/{train,test}/MOT17-XX/`, with
  the images hard-linked rather than copied and the three detection files side
  by side in `det/`;
- exports TrackEval ground truth for the `train`, `train_half` and `val_half`
  splits to `data/mot17/trackeval/`;
- records sequence metadata and split ranges in `data/mot17/manifest.json`.

`data/mot17` is self-contained afterwards, so `MOT17/` can be deleted to free
the duplicate copies (about 3.9 GB).

The half split cuts each training sequence in time: frames `1 .. n//2+1` for
training and the rest for validation, following CenterTrack so that numbers stay
comparable with published ablations. Never split by detector folder, since the
three folders are the same video.

Quirks the loaders handle:

- Boxes are stored as 1-based `left, top, width, height`; loaders return 0-based
  `x1, y1, x2, y2`.
- Some detection files are not sorted by frame.
- DPM detections have ten columns and unbounded scores; FRCNN and SDP have seven
  columns and scores in [0, 1]. Score thresholds are not transferable between
  detectors.
- Ground-truth boxes often extend past the image border; clip before cropping.
- People stay annotated while fully occluded (visibility near 0). Their crops
  show whoever is in front, so appearance training filters by visibility.

To print per-sequence statistics (occlusion gaps, visibility, identity overlap
between the halves):

```bash
uv run python -m reidtrack.data.stats --root data/mot17
```

MOT17 is not part of this repository and remains under its own terms.

## Tests

```bash
uv run pytest
```

## License

Apache-2.0; see [LICENSE](LICENSE).
