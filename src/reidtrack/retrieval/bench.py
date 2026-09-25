"""Latency of candidate retriever backbones, eager vs CUDA-graph replay.

    python -m reidtrack.retrieval.bench

Weights are random: speed does not depend on them.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from reidtrack.report import format_table
from reidtrack.retrieval.graphs import GraphedForward
from reidtrack.retrieval.models import BACKBONES, build_backbone


def _ms(fn, n: int = 40) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n * 1e3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.retrieval.bench", description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    parser.add_argument("--batches", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--size", nargs=2, type=int, default=[256, 128], metavar=("H", "W"))
    parser.add_argument("--channels-last", action="store_true", help="use the channels-last memory layout")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 1

    shape = (3, *args.size)
    layout = torch.channels_last if args.channels_last else torch.contiguous_format
    rows = []
    for name in args.models:
        model = build_backbone(name).eval()
        with FlopCounterMode(display=False) as counter, torch.inference_mode():
            model(torch.zeros(1, *shape))
        gmacs = counter.get_total_flops() / 2e9
        params = sum(p.numel() for p in model.parameters()) / 1e6
        model = model.cuda().half().to(memory_format=layout)
        graphed = GraphedForward(model, shape, buckets=tuple(args.batches), memory_format=layout)
        cells = []
        with torch.inference_mode():
            for b in args.batches:
                x = torch.randn(b, *shape, device="cuda").half().contiguous(memory_format=layout)
                cells += [f"{_ms(lambda: model(x)):.1f}", f"{_ms(lambda: graphed(x)):.1f}"]
        rows.append([name, f"{params:.1f}", f"{gmacs:.2f}", *cells])
        del model, graphed
        torch.cuda.empty_cache()

    headers = ["backbone", "params M", "GMACs/crop"]
    for b in args.batches:
        headers += [f"{b} eager", f"{b} graph"]
    print(
        format_table(
            headers, rows,
            caption=f"Retriever latency, ms per batch, fp16, {args.size[0]}x{args.size[1]}, "
            f"{'channels-last' if args.channels_last else 'NCHW'}, {torch.cuda.get_device_name(0)}",
            note="columns: crops per call, eager vs CUDA-graph replay",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
