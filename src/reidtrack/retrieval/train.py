"""Fine-tune a retriever (appearance model) on MOT17 training-half crops.

    python -m reidtrack.retrieval.train --backbone osnet_x1_0 --init data/weights/osnet_x1_0_msmt17.pth
    python -m reidtrack.retrieval.train --backbone osnet_x1_0 --init ... --score-only
    python -m reidtrack.retrieval.train ... --max-steps 30          # smoke test

Losses follow Luo et al. (Bag of Tricks, 2019): label-smoothed identity
cross-entropy after a BNNeck plus a batch-hard triplet loss before it. Batches
come from ``SameSceneSampler``. Scored on validation-half people never seen in
training (``reidtrack.retrieval.scoring``). Checkpoints go to
data/weights/retriever/<name>/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from reidtrack.report import format_table
from reidtrack.retrieval.augment import Augment, normalize
from reidtrack.retrieval.crops import load_crops
from reidtrack.retrieval.models import BACKBONES, ReIDModel, build_backbone
from reidtrack.retrieval.sampling import Prefetcher, SameSceneSampler
from reidtrack.retrieval.scoring import retrieval_scores


def batch_hard_triplet(features: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """For each anchor, the farthest positive must be closer than the nearest negative by ``margin``."""
    dist = torch.cdist(features.float(), features.float())
    same = labels[:, None] == labels[None, :]
    hardest_positive = (dist * same).max(dim=1).values
    hardest_negative = dist.masked_fill(same, float("inf")).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def build_model(backbone: str, num_classes: int, init: str | None) -> ReIDModel:
    model = ReIDModel(build_backbone(backbone), num_classes)
    if init is None:
        return model
    if init == "imagenet":
        if not backbone.startswith("resnet"):
            raise ValueError("--init imagenet is only wired up for ResNet backbones")
        import torchvision

        weights = getattr(torchvision.models, f"ResNet{backbone.removeprefix('resnet')}_Weights").IMAGENET1K_V1
        reference = getattr(torchvision.models, backbone)(weights=weights).state_dict()
        body_keys = model.backbone.body.state_dict().keys()
        mapping = dict(zip(body_keys, [k for k in reference if not k.startswith("fc.")]))
        model.backbone.body.load_state_dict({new: reference[old] for new, old in mapping.items()})
        return model
    state = torch.load(init, map_location="cpu", weights_only=True)
    if "backbone" in state:  # one of our checkpoints
        state = {k: v for k, v in state["state_dict"].items() if not k.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
    else:  # a torchreid OSNet checkpoint
        state = {k.removeprefix("module."): v for k, v in state.get("state_dict", state).items()}
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        missing, unexpected = model.backbone.net.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(("classifier", "bnneck")) for k in missing):
        raise ValueError(f"{init} does not fit {backbone}: missing {missing}, unexpected {unexpected}")
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reidtrack.retrieval.train", description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", choices=BACKBONES, default="osnet_x1_0")
    parser.add_argument("--init", help="torchreid or reidtrack checkpoint, or 'imagenet' for ResNet")
    parser.add_argument("--name", help="run name (default: <backbone>_mot17)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--people", type=int, default=16)
    parser.add_argument("--crops", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup", type=float, default=2, help="warm-up epochs")
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int, help="stop early (smoke test)")
    parser.add_argument("--score-only", action="store_true", help="score --init without training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--out", type=Path, default=Path("data/weights/retriever"))
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    train, val = load_crops(args.root, "train_half"), load_crops(args.root, "val_half")
    if train.names != val.names:
        raise ValueError("train and val crops list sequences in a different order")
    train_ids = train.identities()
    classes, labels = np.unique(train_ids, return_inverse=True)
    seen = set(classes.tolist())
    device = "cuda"

    model = build_model(args.backbone, len(classes), args.init).to(device)

    def embed(images: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16):
            return model(normalize(images))

    def score() -> dict[str, float]:
        model.eval()
        result = retrieval_scores(embed, val, seen)
        model.train()
        return result

    name = args.name or f"{args.backbone}_mot17"
    if args.score_only:
        s = score()
        print(format_table(["model", "mAP", "Rank-1", "queries"],
                           [[args.init or args.backbone, f"{s['mAP']:.1f}", f"{s['rank1']:.1f}", s["queries"]]],
                           caption="Retrieval on unseen validation-half people"))
        return 0

    sampler = SameSceneSampler(train, args.people, args.crops, seed=args.seed)
    steps_per_epoch = sampler.batches_per_epoch
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup * steps_per_epoch))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, 0.1 + 0.9 * step / warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    scaler = torch.amp.GradScaler("cuda")
    # The scaler skips the first optimiser steps while it calibrates the loss scale,
    # which makes PyTorch warn about the scheduler stepping first. That is expected.
    warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)`")
    augment = Augment()
    labels_t = labels.astype(np.int64)

    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    config = {"backbone": args.backbone, "input_size": list(train.images.shape[2:]), "num_classes": len(classes)}
    history, best, step = [], -1.0, 0
    model.train()
    epoch = 0
    while step < total_steps:
        epoch += 1
        start, id_losses, tri_losses = time.perf_counter(), [], []
        for images, batch_labels in Prefetcher(train, sampler, labels_t):
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            x = augment(images)
            with torch.autocast("cuda", dtype=torch.float16):
                features, logits = model(x)
                id_loss = F.cross_entropy(logits.float(), batch_labels, label_smoothing=0.1)
            tri_loss = batch_hard_triplet(features, batch_labels, args.margin)
            loss = id_loss + tri_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            id_losses.append(id_loss.item())
            tri_losses.append(tri_loss.item())
            step += 1
            if step >= total_steps:
                break
        seconds = time.perf_counter() - start
        row = {"epoch": epoch, "steps": step, "id_loss": float(np.mean(id_losses)),
               "triplet_loss": float(np.mean(tri_losses)), "lr": schedule.get_last_lr()[0], "seconds": seconds}
        if epoch % args.eval_every == 0 or step >= total_steps:
            row.update(score())
            if row["mAP"] > best:
                best = row["mAP"]
                torch.save({**config, "state_dict": model.state_dict(), "epoch": epoch, "scores": row}, out / "best.pt")
        history.append(row)
        scored = f"  mAP {row['mAP']:.1f}  R1 {row['rank1']:.1f}" if "mAP" in row else ""
        print(f"epoch {epoch:3d}  step {step:6d}  id {row['id_loss']:.3f}  tri {row['triplet_loss']:.3f}  "
              f"lr {row['lr']:.2e}  {seconds:.0f} s{scored}", flush=True)
    torch.save({**config, "state_dict": model.state_dict(), "epoch": epoch}, out / "last.pt")
    (out / "history.json").write_text(json.dumps({"args": vars(args) | {"root": str(args.root), "out": str(args.out)},
                                                  "history": history}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
