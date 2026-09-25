"""Fine-tune a retriever (appearance model) on MOT17 training-half crops.

    python -m reidtrack.retrieval.train --backbone osnet_x1_0 --init data/weights/osnet_x1_0_msmt17.pth
    python -m reidtrack.retrieval.train --backbone osnet_x1_0 --init ... --score-only
    python -m reidtrack.retrieval.train ... --max-steps 30          # smoke test
    python -m reidtrack.retrieval.train --backbone osnet_x1_0 --resume  # continue after a pause

Losses follow Luo et al. (Bag of Tricks, 2019): label-smoothed identity
cross-entropy after a BNNeck plus a batch-hard triplet loss before it. Batches
come from ``SameSceneSampler``. Scored on validation-half people never seen in
training (``reidtrack.retrieval.scoring``). Checkpoints go to
data/weights/retriever/<name>/.

Ctrl+C, or a file named PAUSE in the run folder, pauses the run at the next step:
its full state is saved to resume.pt and --resume continues it. The state is
also saved at every epoch end, so an interrupted run loses at most one epoch.
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

from reidtrack.pausing import PAUSED, PauseRequest
from reidtrack.report import format_table
from reidtrack.retrieval.augment import GROUPS, augment_groups, normalize
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


TRAINING_KEYS = ("backbone", "init", "epochs", "people", "crops", "lr", "weight_decay", "warmup", "margin", "augment",
                 "eval_every", "max_steps", "seed", "sequences")


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
    parser.add_argument("--resume", action="store_true", help="continue a paused or interrupted run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", default=",".join(GROUPS), help=f"augmentation groups, comma-separated, from {', '.join(GROUPS)}")
    parser.add_argument("--sequences", default="", help="train only on these sequences, comma-separated (cross-fitting)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--out", type=Path, default=Path("data/weights/retriever"))
    args = parser.parse_args(argv)

    name = args.name or f"{args.backbone}_mot17"
    out = args.out / name
    resume_path = out / "resume.pt"
    saved = None
    if args.resume:
        if not resume_path.is_file():
            print(f"nothing to resume: {resume_path} not found", file=sys.stderr)
            return 1
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key in TRAINING_KEYS:  # a resumed run keeps its original schedule
            setattr(args, key, saved["args"].get(key, getattr(args, key)))

    torch.manual_seed(args.seed)
    train, val = load_crops(args.root, "train_half"), load_crops(args.root, "val_half")
    if train.names != val.names:
        raise ValueError("train and val crops list sequences in a different order")
    chosen = [s for s in args.sequences.split(",") if s]
    unknown = sorted(set(chosen) - set(train.names))
    if unknown:
        print(f"unknown sequences: {', '.join(unknown)}", file=sys.stderr)
        return 1
    allowed = np.isin(train.sequence, [train.names.index(s) for s in chosen]) if chosen else np.ones(len(train), bool)
    classes, inverse = np.unique(train.identities()[allowed], return_inverse=True)
    labels = np.full(len(train), -1, dtype=np.int64)
    labels[allowed] = inverse
    seen = set(classes.tolist())
    device = args.device
    amp = device.startswith("cuda")

    model = build_model(args.backbone, len(classes), args.init).to(device)

    def embed(images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=amp):
            return model(normalize(images))

    def score() -> dict[str, float]:
        model.eval()
        result = retrieval_scores(embed, val, seen, device=device)
        model.train()
        return result

    if args.score_only:
        s = score()
        print(format_table(["model", "mAP", "Rank-1", "queries"],
                           [[args.init or args.backbone, f"{s['mAP']:.1f}", f"{s['rank1']:.1f}", s["queries"]]],
                           caption="Retrieval on unseen validation-half people"))
        return 0

    sampler = SameSceneSampler(train, args.people, args.crops, seed=args.seed,
                               sequences={train.names.index(s) for s in chosen} if chosen else None)
    steps_per_epoch = sampler.batches_per_epoch
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup * steps_per_epoch))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                 weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, 0.1 + 0.9 * step / warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=amp)
    # The scaler skips the first optimiser steps while it calibrates the loss scale,
    # which makes PyTorch warn about the scheduler stepping first. That is expected.
    warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)`")
    augment = augment_groups(tuple(g for g in args.augment.split(",") if g))
    labels_t = labels.astype(np.int64)

    out.mkdir(parents=True, exist_ok=True)
    config = {"backbone": args.backbone, "input_size": list(train.images.shape[2:]), "num_classes": len(classes)}
    run_args = {k: getattr(args, k) for k in TRAINING_KEYS}
    progress = {"step": 0, "epoch": 0, "epoch_step": 0, "best": -1.0, "history": [],
                "id_losses": [], "tri_losses": [], "seconds": 0.0}
    if saved is not None:
        if saved["config"] != config:
            print(f"{resume_path} was made for {saved['config']}, not {config}", file=sys.stderr)
            return 1
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        schedule.load_state_dict(saved["schedule"])
        torch.set_rng_state(saved["torch_rng"])
        sampler.rng.bit_generator.state = saved["sampler_rng"]
        progress = saved["progress"]
        print(f"resuming {name} at epoch {progress['epoch']}, step {progress['step']} of {total_steps}", flush=True)

    def save_resume() -> None:
        tmp = resume_path.with_suffix(".tmp")
        torch.save({"config": config, "args": run_args, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "schedule": schedule.state_dict(), "torch_rng": torch.get_rng_state(),
                    "sampler_rng": sampler.rng.bit_generator.state, "progress": progress}, tmp)
        tmp.replace(resume_path)  # atomic: a crash mid-save keeps the previous state

    pause = PauseRequest(out)
    model.train()
    try:
        while progress["step"] < total_steps:
            if progress["epoch_step"] == 0:
                progress.update(epoch=progress["epoch"] + 1, id_losses=[], tri_losses=[], seconds=0.0)
            remaining = min(steps_per_epoch - progress["epoch_step"], total_steps - progress["step"])
            start = time.perf_counter()
            for images, batch_labels in Prefetcher(train, sampler, labels_t, batches=remaining):
                images = images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                x = augment(images)
                with torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=amp):
                    features, logits = model(x)
                    id_loss = F.cross_entropy(logits.float(), batch_labels, label_smoothing=0.1)
                tri_loss = batch_hard_triplet(features, batch_labels, args.margin)
                loss = id_loss + tri_loss
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                schedule.step()
                progress["id_losses"].append(id_loss.item())
                progress["tri_losses"].append(tri_loss.item())
                progress["step"] += 1
                progress["epoch_step"] += 1
                if pause:
                    break
            progress["seconds"] += time.perf_counter() - start
            epoch_done = progress["epoch_step"] >= steps_per_epoch or progress["step"] >= total_steps
            if epoch_done:
                row = {"epoch": progress["epoch"], "steps": progress["step"],
                       "id_loss": float(np.mean(progress["id_losses"])),
                       "triplet_loss": float(np.mean(progress["tri_losses"])),
                       "lr": schedule.get_last_lr()[0], "seconds": progress["seconds"]}
                if progress["epoch"] % args.eval_every == 0 or progress["step"] >= total_steps:
                    row.update(score())
                    if row["mAP"] > progress["best"]:
                        progress["best"] = row["mAP"]
                        torch.save({**config, "state_dict": model.state_dict(), "epoch": progress["epoch"],
                                    "scores": row}, out / "best.pt")
                progress["history"].append(row)
                progress["epoch_step"] = 0
                scored = f"  mAP {row['mAP']:.1f}  R1 {row['rank1']:.1f}" if "mAP" in row else ""
                print(f"epoch {row['epoch']:3d}  step {row['steps']:6d}  id {row['id_loss']:.3f}  "
                      f"tri {row['triplet_loss']:.3f}  lr {row['lr']:.2e}  {row['seconds']:.0f} s{scored}", flush=True)
            if progress["step"] < total_steps:
                save_resume()  # every epoch end, and on pause
                if pause:
                    print(f"paused at epoch {progress['epoch']}, step {progress['step']} of {total_steps}; "
                          f"continue with the same command plus --resume", flush=True)
                    return PAUSED
    finally:
        pause.close()

    torch.save({**config, "state_dict": model.state_dict(), "epoch": progress["epoch"]}, out / "last.pt")
    (out / "history.json").write_text(json.dumps(
        {"args": run_args | {"root": str(args.root), "out": str(args.out)}, "history": progress["history"]},
        indent=2) + "\n")
    resume_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
