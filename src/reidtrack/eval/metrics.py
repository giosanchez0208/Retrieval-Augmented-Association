"""Score tracker output on a MOT17 split with TrackEval (HOTA, CLEAR, Identity)."""

from __future__ import annotations

import contextlib
import io
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from reidtrack.data.mot import Tracks, save_tracks
from reidtrack.data.mot17 import Mot17


@dataclass(frozen=True)
class Scores:
    hota: float
    deta: float
    assa: float
    mota: float
    idf1: float
    idsw: int
    fp: int
    fn: int

    HEADERS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW")

    @classmethod
    def from_trackeval(cls, res: dict) -> Scores:
        hota, clear, identity = res["HOTA"], res["CLEAR"], res["Identity"]
        return cls(
            hota=100 * float(np.mean(hota["HOTA"])),
            deta=100 * float(np.mean(hota["DetA"])),
            assa=100 * float(np.mean(hota["AssA"])),
            mota=100 * float(clear["MOTA"]),
            idf1=100 * float(identity["IDF1"]),
            idsw=int(clear["IDSW"]),
            fp=int(clear["CLR_FP"]),
            fn=int(clear["CLR_FN"]),
        )

    def row(self) -> list[str]:
        return [f"{v:.1f}" for v in (self.hota, self.deta, self.assa, self.mota, self.idf1)] + [str(self.idsw)]


@dataclass(frozen=True)
class Evaluation:
    split: str
    sequences: dict[str, Scores]
    combined: Scores

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "sequences": {name: asdict(s) for name, s in self.sequences.items()},
            "combined": asdict(self.combined),
        }


def split_sequences(split: str, root: str | Path = "data/mot17") -> list[str]:
    seqmap = Path(root) / "trackeval" / "seqmaps" / f"MOT17-{split}.txt"
    if not seqmap.is_file():
        raise FileNotFoundError(f"{seqmap} not found; run `python -m reidtrack.data.prepare` first")
    return seqmap.read_text().split()[1:]


def evaluate(results: Mapping[str, Tracks], split: str = "val_half", root: str | Path = "data/mot17") -> Evaluation:
    """Score tracks against the ground truth of ``split``.

    Tracks use the sequence's own frame numbers; rows outside the split are
    dropped and the rest renumbered to match the split's ground truth.
    """
    root = Path(root)
    names = split_sequences(split, root)
    missing = sorted(set(names) - set(results))
    if missing:
        raise KeyError(f"no results for {', '.join(missing)}")

    trackeval = _import_trackeval()
    data = Mot17(root)
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            rng = data.sequence(name).frames(split)
            tracks = results[name].in_frames(rng.first, rng.last)
            save_tracks(Path(tmp) / "tracker" / f"{name}.txt", replace(tracks, frame=tracks.frame - rng.offset))

        dataset_config = {
            **trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
            "GT_FOLDER": str(root / "trackeval" / f"MOT17-{split}"),
            "SEQMAP_FILE": str(root / "trackeval" / "seqmaps" / f"MOT17-{split}.txt"),
            "SKIP_SPLIT_FOL": True,
            "TRACKERS_FOLDER": tmp,
            "TRACKERS_TO_EVAL": ["tracker"],
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_FOLDER": tmp,
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": split,
            "CLASSES_TO_EVAL": ["pedestrian"],
            "PRINT_CONFIG": False,
        }
        eval_config = {
            **trackeval.Evaluator.get_default_eval_config(),
            "USE_PARALLEL": False,
            "BREAK_ON_ERROR": True,
            "LOG_ON_ERROR": None,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
            evaluator = trackeval.Evaluator(eval_config)
            output, _ = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)

    res = output["MotChallenge2DBox"]["tracker"]
    return Evaluation(
        split=split,
        sequences={name: Scores.from_trackeval(res[name]["pedestrian"]) for name in names},
        combined=Scores.from_trackeval(res["COMBINED_SEQ"]["pedestrian"]),
    )


def _import_trackeval():
    # TrackEval's MOT17 loader and metrics still use np.float / np.int / np.bool,
    # which numpy removed in 1.24. They were plain aliases of the builtins.
    for name, builtin in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, builtin)
    # Importing prints a notice about optional loaders (BURST) we don't use.
    with contextlib.redirect_stdout(io.StringIO()):
        import trackeval

    return trackeval
