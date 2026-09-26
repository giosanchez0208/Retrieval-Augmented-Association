"""The whole tracker on raw frames: decode, detect, crop and embed, track, timed per frame.

    python -m reidtrack.pipeline MOT17-08

Frames are decoded on the GPU and stay there for the detector and the appearance model.
The detector runs in float16 as a TorchScript graph, and only boxes confident enough for
the tracker to use their appearance get embedded.
Writes the tracks in MOT format and a CSV of each frame's stage times (ms) to
runs/pipeline_<sequence>/, which ``reidtrack.viz --timing`` can print under each frame.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import roi_align

STAGES = ("decode", "detect", "embed", "track")


class Pipeline:
    def __init__(self, detector: Path, embedder: Path, reranker: Path, width: int, height: int, frame_rate: float,
                 min_score: float = 0.1, fast_detector: bool = True) -> None:
        from rfdetr import RFDETRSmall

        from reidtrack.association.reranker import load_reranker
        from reidtrack.retrieval.embedder import Embedder
        from reidtrack.retrieval.graphs import GraphedForward
        from reidtrack.tracker import RetrievalTracker, TrackerConfig

        self.detector = RFDETRSmall(pretrain_weights=str(detector), device="cuda")
        if fast_detector:  # 27.7 -> 12.7 ms on MOT17-08, with 99.1% overlap between the boxes
            self.detector.inference(compile=True, dtype=torch.float16, batch_size=1)
        self.min_score, self.person = min_score, None
        embed = Embedder(embedder)
        self.decode_jpeg, self.mean, self.std = embed.decode, embed.mean, embed.std
        self.input_size = embed.input_size
        self.model = embed.model.half()
        self.graphed = GraphedForward(self.model, (3, *self.input_size), buckets=(8, 16, 32, 64))
        self.dim = embed.dim
        self.tracker = RetrievalTracker(width, height, frame_rate, TrackerConfig(), load_reranker(reranker))

    @torch.inference_mode()
    def _embed(self, image: torch.Tensor, xyxy: np.ndarray) -> np.ndarray:
        if len(xyxy) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        h, w = image.shape[1:]
        boxes = torch.as_tensor(np.asarray(xyxy, dtype=np.float32), device=image.device)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        rois = torch.cat([boxes.new_zeros(len(boxes), 1), boxes], dim=1)
        crops = roi_align(image[None].float(), rois, self.input_size, aligned=True)
        crops = ((crops / 255 - self.mean) / self.std).half()
        return torch.nn.functional.normalize(self.graphed(crops).float(), dim=1).cpu().numpy()

    def __call__(self, jpeg: bytes) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
        """One frame's tracks (ids, xyxy, scores) and its stage times in ms."""
        times, start = {}, time.perf_counter()

        def mark(stage: str) -> None:
            nonlocal start
            torch.cuda.synchronize()
            now = time.perf_counter()
            times[stage] = (now - start) * 1e3
            start = now

        image = self.decode_jpeg(jpeg)
        mark("decode")
        d = self.detector.predict(image.float() / 255, threshold=self.min_score)
        if self.person is None and len(d):  # the fine-tuned model has one class; find its id once
            self.person = Counter(d.class_id.tolist()).most_common(1)[0][0]
        keep = d.class_id == self.person
        xyxy, scores = d.xyxy[keep].astype(np.float64), d.confidence[keep].astype(np.float64)
        mark("detect")
        # the tracker only reads appearance for boxes at or above high_score, so skip the rest
        feats = np.zeros((len(xyxy), self.dim), dtype=np.float32)
        confident = scores >= self.tracker.cfg.high_score
        feats[confident] = self._embed(image, xyxy[confident])
        mark("embed")
        out = self.tracker.update(xyxy, scores, feats)
        mark("track")
        return out, times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.pipeline", description=__doc__.split("\n\n")[0])
    parser.add_argument("sequence", help="a MOT17 video, training or test, e.g. MOT17-08")
    parser.add_argument("--detector", type=Path,
                        default=Path("data/weights/detectors/rf-detr-small-mot17/checkpoint_best_total.pth"))
    parser.add_argument("--embedder", type=Path, default=Path("data/weights/retriever/osnet_x0_5_mot17/last.pt"))
    parser.add_argument("--reranker", type=Path, default=Path("data/weights/reranker/pairwise_osnet_x0_5_iou_last.pt"))
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--out", type=Path, help="default: runs/pipeline_<sequence>")
    parser.add_argument("--plain-detector", action="store_true", help="run the detector in float32 without compiling")
    args = parser.parse_args(argv)

    from reidtrack.data.mot import Tracks, save_tracks
    from reidtrack.data.mot17 import Mot17

    seq = Mot17(args.root).sequence(args.sequence)
    pipe = Pipeline(args.detector, args.embedder, args.reranker, seq.info.width, seq.info.height, seq.info.frame_rate,
                    fast_detector=not args.plain_detector)
    out = args.out or Path("runs") / f"pipeline_{seq.name}"
    out.mkdir(parents=True, exist_ok=True)
    rows, frames, ids, boxes, scores = [], [], [], [], []
    for frame in range(1, seq.info.length + 1):
        (i, b, s), times = pipe(seq.image_path(frame).read_bytes())
        rows.append({"frame": frame, **{k: round(times[k], 2) for k in STAGES}, "total": round(sum(times.values()), 2)})
        frames.append(np.full(len(i), frame, dtype=np.int32)); ids.append(i); boxes.append(b); scores.append(s)
    save_tracks(out / f"{seq.name}.txt", Tracks(frame=np.concatenate(frames), track_id=np.concatenate(ids).astype(np.int32),
                                                xyxy=np.concatenate(boxes).astype(np.float32).reshape(-1, 4),
                                                score=np.concatenate(scores).astype(np.float32)))
    with open(out / "timing.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", *STAGES, "total"])
        writer.writeheader()
        writer.writerows(rows)
    steady = rows[10:]  # the first frames include warm-up
    means = {k: np.mean([r[k] for r in steady]) for k in (*STAGES, "total")}
    print(f"{seq.name}: {len(np.unique(np.concatenate(ids)))} IDs over {seq.info.length} frames; ms per frame after "
          f"warm-up: " + ", ".join(f"{k} {v:.1f}" for k, v in means.items()) + f" ({1e3 / means['total']:.0f} fps) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
