"""Fine-tune RF-DETR on the MOT17 person dataset, stopping once it plateaus.

    python -m reidtrack.detection.coco
    python -m reidtrack.detection.finetune

Training stops after ``--patience`` epochs without improvement on the ``valid`` folder,
the last quarter of each video's training half, or after ``--epochs`` at most.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.detection.finetune", description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=Path("data/mot17/coco"))
    parser.add_argument("--weights", type=Path, default=Path("data/weights/detectors/rf-detr-small.pth"))
    parser.add_argument("--out", type=Path, default=Path("data/weights/detectors/rf-detr-small-mot17"))
    parser.add_argument("--epochs", type=int, default=30, help="upper bound; early stopping usually ends sooner")
    parser.add_argument("--patience", type=int, default=3, help="epochs without improvement before stopping")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args(argv)

    from rfdetr import RFDETRSmall

    model = RFDETRSmall(pretrain_weights=str(args.weights))
    model.train(dataset_dir=str(args.dataset), output_dir=str(args.out), epochs=args.epochs, batch_size=args.batch,
                grad_accum_steps=args.accum, lr=args.lr, early_stopping=True, early_stopping_patience=args.patience)
    return 0


if __name__ == "__main__":
    sys.exit(main())
