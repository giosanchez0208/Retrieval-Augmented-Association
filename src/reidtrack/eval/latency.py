"""Per-stage wall-clock timing for the tracking pipeline.

    timer = StageTimer(sync=torch.cuda.synchronize)
    for frame in frames:
        with timer.stage("detect"):
            ...
        timer.next_frame()
    print(timer.table())

Run as a module to benchmark the input path (read, JPEG decode, upload):

    python -m reidtrack.eval.latency --seq MOT17-04
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from reidtrack.report import format_table


class StageTimer:
    """Accumulates per-stage times per frame; the first ``warmup`` frames are discarded.

    ``sync`` runs before and after every stage, e.g. ``torch.cuda.synchronize`` so
    asynchronous GPU work is charged to the stage that launched it.
    """

    def __init__(
        self,
        warmup: int = 10,
        sync: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.warmup = warmup
        self.sync = sync
        self.clock = clock
        self._seen = 0
        self._names: list[str] = []
        self._frames: list[tuple[dict[str, float], float]] = []
        self._current: dict[str, float] = {}
        self._frame_start: float | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.sync:
            self.sync()
        start = self.clock()
        if self._frame_start is None:
            self._frame_start = start
        try:
            yield
        finally:
            if self.sync:
                self.sync()
            self._current[name] = self._current.get(name, 0.0) + self.clock() - start
            if name not in self._names:
                self._names.append(name)

    def next_frame(self) -> None:
        end = self.clock()
        if self._frame_start is not None and self._seen >= self.warmup:
            self._frames.append((self._current, end - self._frame_start))
        self._seen += 1
        self._current = {}
        self._frame_start = None

    @property
    def frames(self) -> int:
        return len(self._frames)

    def summary(self) -> dict[str, dict[str, float]]:
        """Milliseconds per frame for each stage, untimed overhead ("other") and the total."""
        if not self._frames:
            return {}
        per_stage = {n: np.array([f.get(n, 0.0) for f, _ in self._frames]) * 1e3 for n in self._names}
        total = np.array([t for _, t in self._frames]) * 1e3
        per_stage["other"] = np.clip(total - sum(per_stage.values()), 0, None)
        per_stage["total"] = total
        return {
            name: {
                "mean": float(ms.mean()),
                "p50": float(np.percentile(ms, 50)),
                "p95": float(np.percentile(ms, 95)),
                "share": float(ms.mean() / total.mean()) if total.mean() > 0 else 0.0,
            }
            for name, ms in per_stage.items()
        }

    def fps(self) -> float:
        s = self.summary()
        return 1e3 / s["total"]["mean"] if s and s["total"]["mean"] > 0 else 0.0

    def table(self, caption: str | None = None) -> str:
        s = self.summary()

        def row(name: str) -> list[str]:
            v = s[name]
            return [name, f"{v['mean']:.2f}", f"{v['p50']:.2f}", f"{v['p95']:.2f}", f"{100 * v['share']:.0f}%"]

        return format_table(
            ["stage", "mean ms", "p50", "p95", "share"],
            [row(n) for n in s if n != "total"],
            caption=caption or f"{self.frames} frames, {self.fps():.1f} fps",
            footer=[row("total")],
        )


def _decoders(use_cuda: bool) -> dict[str, Callable[[bytes], object]]:
    import cv2

    decoders: dict[str, Callable[[bytes], object]] = {
        "cv2": lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR),
        "cv2 1/2": lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_REDUCED_COLOR_2),
    }
    if use_cuda:
        import torch
        from torchvision.io import decode_jpeg

        def nvjpeg(b: bytes) -> object:
            return decode_jpeg(torch.frombuffer(bytearray(b), dtype=torch.uint8), device="cuda")

        decoders["nvjpeg"] = nvjpeg
    return decoders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reidtrack.eval.latency", description="Benchmark the frame input path."
    )
    parser.add_argument("--root", type=Path, default=Path("data/mot17"), help="prepared dataset root")
    parser.add_argument("--seq", default="MOT17-04")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="skip GPU upload and GPU decoding")
    args = parser.parse_args(argv)

    from reidtrack.data.mot17 import Mot17

    seq = Mot17(args.root).sequence(args.seq)
    paths = [seq.image_path(f) for f in range(1, min(args.frames, seq.info.length) + 1)]
    for path in paths:  # warm the OS file cache so every decoder sees the same conditions
        path.read_bytes()

    use_cuda = False
    if not args.cpu:
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            pass
    sync = None
    if use_cuda:
        import torch

        sync = torch.cuda.synchronize

    rows = []
    for name, decode in _decoders(use_cuda).items():
        timer = StageTimer(warmup=args.warmup, sync=sync)
        try:
            for path in paths:
                with timer.stage("read"):
                    data = path.read_bytes()
                with timer.stage("decode"):
                    image = decode(data)
                if use_cuda and name != "nvjpeg":
                    with timer.stage("upload"):
                        torch.from_numpy(image).to("cuda")
                timer.next_frame()
        except RuntimeError as err:
            print(f"{name}: skipped ({err})", file=sys.stderr)
            continue
        s = timer.summary()
        upload = f"{s['upload']['mean']:.2f}" if "upload" in s else "-"
        rows.append(
            [name, f"{s['read']['mean']:.2f}", f"{s['decode']['mean']:.2f}", upload,
             f"{s['total']['mean']:.2f}", f"{timer.fps():.0f}"]
        )

    device = torch.cuda.get_device_name(0) if use_cuda else "CPU only"
    print(
        format_table(
            ["decoder", "read", "decode", "upload", "total ms", "fps"],
            rows,
            caption=f"Input path, {seq.name} ({seq.info.width}x{seq.info.height}), "
            f"{len(paths) - args.warmup} frames, {device}",
            note="mean ms per frame, warm file cache; cv2 1/2 decodes at half resolution",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
