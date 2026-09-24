"""Score tracker output on a MOT17 split.

    python -m reidtrack.eval --results runs/sort --split val_half
    python -m reidtrack.eval --oracle              # ground truth as output; must score 100

Result files are named <sequence>.txt, in MOT format, with the sequence's own
frame numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reidtrack.data.mot import Tracks, load_tracks
from reidtrack.data.mot17 import Mot17
from reidtrack.data.splits import SPLITS
from reidtrack.eval.metrics import Scores, evaluate, split_sequences
from reidtrack.report import format_table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.eval", description="Score tracker output on MOT17.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, help="folder of <sequence>.txt result files")
    source.add_argument("--oracle", action="store_true", help="score the ground truth itself")
    parser.add_argument("--split", choices=SPLITS, default="val_half")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"), help="prepared dataset root")
    parser.add_argument("--json", type=Path, help="also write scores to this file")
    args = parser.parse_args(argv)

    names = split_sequences(args.split, args.root)
    if args.oracle:
        data = Mot17(args.root)
        results = {n: Tracks.from_gt(data.sequence(n).load_gt().targets()) for n in names}
        label = "ground truth"
    else:
        missing = [n for n in names if not (args.results / f"{n}.txt").is_file()]
        if missing:
            print(f"missing result files in {args.results}: {', '.join(missing)}", file=sys.stderr)
            return 1
        results = {n: load_tracks(args.results / f"{n}.txt") for n in names}
        label = args.results.name

    ev = evaluate(results, args.split, args.root)
    print(
        format_table(
            ["sequence", *Scores.HEADERS],
            [[name, *s.row()] for name, s in ev.sequences.items()],
            caption=f"MOT17 {args.split}, {label}",
            footer=[["combined", *ev.combined.row()]],
        )
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"source": label, **ev.to_dict()}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
