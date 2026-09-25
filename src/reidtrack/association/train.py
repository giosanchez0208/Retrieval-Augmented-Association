"""Collect candidate pairs from the tracker and train the learned reranker.

    python -m reidtrack.association.train --embeddings osnet_x1_0_msmt17 --collect-only
    python -m reidtrack.association.train --embeddings osnet_x1_0_msmt17

Pairs are recorded by running the hand-set tracker over the training half and
logging every (bank entry, confident detection) pair it considers. A pair is
positive when the detection matches a ground-truth person (IoU >= 0.5) and that
person is who the entry has mostly been matched to so far. This imitates the
tracker's real situations, including its own drift, rather than clean ground truth.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from reidtrack.data.mot17 import Mot17
from reidtrack.eval.metrics import split_sequences
from reidtrack.report import format_table
from reidtrack.track.assignment import linear_assignment
from reidtrack.track.boxes import iou_matrix


def ground_truth_ids(det_xyxy: np.ndarray, gt_xyxy: np.ndarray, gt_ids: np.ndarray, min_iou: float = 0.5) -> np.ndarray:
    """The ground-truth id of each detection, or -1."""
    out = np.full(len(det_xyxy), -1, dtype=np.int64)
    if len(det_xyxy) and len(gt_xyxy):
        iou = iou_matrix(det_xyxy, gt_xyxy)
        pairs, _, _ = linear_assignment(1 - iou, 1 - min_iou)
        for d, g in pairs:
            out[d] = gt_ids[g]
    return out


def collect_frames(split: str, embeddings: str, root: Path, camera: bool = False, detector: str = "FRCNN",
                   reranker=None):
    """Run the tracker over ``split``; one (cues (N, M, C), labels (N, M), sequence) per frame.

    With ``reranker`` the learned matcher drives the tracker while pairs are recorded, so
    the pairs come from the situations that matcher gets itself into (on-policy).
    Without it, the hand-set rules drive."""
    from reidtrack.retrieval.cache import load_embeddings
    from reidtrack.track.camera import load_warps
    from reidtrack.tracker import RetrievalTracker, TrackerConfig

    data = Mot17(root)
    frames = []
    for s, name in enumerate(split_sequences(split, root)):
        seq = data.sequence(name)
        rng = seq.frames(split)
        det = seq.load_det(detector)
        keep = (det.frame >= rng.first) & (det.frame <= rng.last)
        feats = load_embeddings(root, embeddings, detector, name, det.frame)[keep]
        det = det.select(keep)
        gt = seq.load_gt(split).targets()
        warps = load_warps(root, name) if camera else None
        votes: dict[int, Counter] = defaultdict(Counter)
        frame_state = {}

        def record(track_ids, dets, cues):
            frame_state["pairs"] = (track_ids, dets, cues)

        tracker = RetrievalTracker(seq.info.width, seq.info.height, seq.info.frame_rate, TrackerConfig(), reranker,
                                   recorder=record)
        bounds = np.searchsorted(det.frame, np.arange(rng.first, rng.last + 2))
        gbounds = np.searchsorted(gt.frame, np.arange(rng.first, rng.last + 2))
        for i, frame in enumerate(range(rng.first, rng.last + 1)):
            lo, hi = bounds[i], bounds[i + 1]
            glo, ghi = gbounds[i], gbounds[i + 1]
            gt_of = ground_truth_ids(det.xyxy[lo:hi], gt.xyxy[glo:ghi], gt.track_id[glo:ghi])
            frame_state.clear()
            kwargs = {} if warps is None else {"warp": warps[frame - 1]}
            tracker.update(det.xyxy[lo:hi].astype(np.float64), det.score[lo:hi], feats[lo:hi], **kwargs)
            if "pairs" in frame_state:
                track_ids, dets, cues = frame_state["pairs"]
                if len(track_ids) and len(dets):
                    identity = np.array([votes[t].most_common(1)[0][0] if votes[t] else -2 for t in track_ids])
                    det_ids = gt_of[dets]
                    labels = (identity[:, None] == det_ids[None, :]) & (det_ids[None, :] >= 0)
                    frames.append((cues, labels, s))
            for track_id, j in tracker.assignments.items():  # update after labelling: no peeking
                if gt_of[j] >= 0:
                    votes[track_id][int(gt_of[j])] += 1
    return frames


def collect(split: str, embeddings: str, root: Path, camera: bool = False, detector: str = "FRCNN"):
    """Flattened pairs: (cues (P, C), labels (P,), sequence (P,))."""
    frames = collect_frames(split, embeddings, root, camera, detector)
    return (
        np.concatenate([c.reshape(-1, c.shape[-1]) for c, _, _ in frames]),
        np.concatenate([y.reshape(-1) for _, y, _ in frames]),
        np.concatenate([np.full(y.size, s, dtype=np.int16) for _, y, s in frames]),
    )


def pair_scores(model, cues: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from reidtrack.retrieval.scoring import average_precision

    prob = model(cues)
    pred = prob >= 0.5
    return {
        "AP": 100 * average_precision(prob, labels),
        "precision": 100 * float(labels[pred].mean()) if pred.any() else 0.0,
        "recall": 100 * float(pred[labels].mean()) if labels.any() else 0.0,
    }


def train(cues: np.ndarray, labels: np.ndarray, epochs: int, batch: int = 8192, lr: float = 2e-3, seed: int = 0,
          dropped: tuple[str, ...] = ()):
    from reidtrack.association.features import NAMES
    from reidtrack.association.reranker import PairwiseReranker

    torch.manual_seed(seed)
    model = PairwiseReranker(NAMES[: cues.shape[-1]], dropped=dropped)
    x = torch.from_numpy(cues.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    pos_weight = torch.tensor((1 - labels.mean()) / max(labels.mean(), 1e-9))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for i in range(0, len(x), batch):
            idx = order[i : i + batch]
            loss = F.binary_cross_entropy_with_logits(model.forward(x[idx]), y[idx], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def train_context(frames, epochs: int, frames_per_step: int = 8, lr: float = 1e-3, seed: int = 0, device: str = "cpu",
                  dropped: tuple[str, ...] = ()):
    """Train ``AxialReranker`` on whole frames, so each pair is scored among its competitors.

    Frames are padded into batches of ``frames_per_step``; padding is masked out of the
    attention and the loss.
    """
    from reidtrack.association.reranker import AxialReranker, pad_frames

    torch.manual_seed(seed)
    model = AxialReranker(dropped=dropped).to(device)
    rate = np.concatenate([y.reshape(-1) for _, y, _ in frames]).mean()
    pos_weight = torch.tensor((1 - rate) / max(rate, 1e-9), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    data = [(torch.from_numpy(c.astype(np.float32)), torch.from_numpy(y.astype(np.float32))) for c, y, _ in frames]
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(data))
        for i in range(0, len(order), frames_per_step):
            cues, labels, entry_pad, det_pad, valid = (
                t.to(device) for t in pad_frames([data[k] for k in order[i : i + frames_per_step]])
            )
            logits = model(cues, entry_pad, det_pad)
            loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.cpu().eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.association.train", description=__doc__.split("\n\n")[0])
    parser.add_argument("--embeddings", default="osnet_x1_0_msmt17", help="feature cache name")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--model", choices=("pairwise", "context"), default="pairwise",
                        help="pairwise: each pair alone; context: pairs attend to their competitors")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cpu", help="the context model trains faster on cuda")
    parser.add_argument("--drop-cues", default="", help="comma-separated cues to zero, e.g. sim_best,sim_average")
    parser.add_argument("--name", help="checkpoint name (default: <model>_<embeddings>)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="rounds after the first record pairs with the previous round's model driving the tracker")
    parser.add_argument("--collect-only", action="store_true", help="only record pairs and report their statistics")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--out", type=Path, default=Path("data/weights/reranker"))
    args = parser.parse_args(argv)
    dropped = tuple(c for c in args.drop_cues.split(",") if c)

    frames = collect_frames("train_half", args.embeddings, args.root, args.camera)
    names = split_sequences("train_half", args.root)
    labels = np.concatenate([y.reshape(-1) for _, y, _ in frames])
    print(f"collected {len(frames):,} frames, {len(labels):,} pairs, {int(labels.sum()):,} positive "
          f"({100 * labels.mean():.2f}%)")
    if args.collect_only:
        return 0

    if args.model == "pairwise":
        def fit(fs):
            return train(np.concatenate([c.reshape(-1, c.shape[-1]) for c, _, _ in fs]),
                         np.concatenate([y.reshape(-1) for _, y, _ in fs]), args.epochs, dropped=dropped)
    else:
        def fit(fs):
            return train_context(fs, args.epochs, device=args.device, dropped=dropped)
    model = fit(frames)
    for r in range(2, args.rounds + 1):  # on-policy: the matcher drives, its own situations get labelled
        more = collect_frames("train_half", args.embeddings, args.root, args.camera, reranker=model)
        frames += more
        print(f"round {r}: +{len(more):,} frames, {sum(y.size for _, y, _ in more):,} pairs")
        model = fit(frames)

    # sanity check on held-out sequences. After round 1 their later pairs were recorded by a
    # model that had trained on them, so the check gets looser with each round.
    held_seqs = {names.index("MOT17-09"), names.index("MOT17-11")}
    kept = [f for f in frames if f[2] not in held_seqs]
    held = [f for f in frames if f[2] in held_seqs]
    probe = fit(kept)
    held_prob = np.concatenate([probe(c).reshape(-1) for c, _, _ in held])
    held_labels = np.concatenate([y.reshape(-1) for _, y, _ in held])
    s = pair_scores(lambda _: held_prob, None, held_labels)
    out = args.out / f"{args.name or args.model + '_' + args.embeddings}{'_camera' if args.camera else ''}.pt"
    model.save(out, embeddings=args.embeddings, epochs=args.epochs, rounds=args.rounds,
               pairs=int(sum(y.size for _, y, _ in frames)))
    print(format_table(["check", "AP", "precision", "recall"],
                       [["held-out MOT17-09, MOT17-11", f"{s['AP']:.1f}", f"{s['precision']:.1f}", f"{s['recall']:.1f}"]],
                       caption=f"{args.model.capitalize()} reranker, same-person classification at p >= 0.5",
                       note=f"final model trained on all training-half pairs: {out}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
