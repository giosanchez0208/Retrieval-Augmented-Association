"""Retrieval accuracy of an appearance model on held-out people.

Queries are crops of validation-half people who never appear in the training
half. Each query searches the crops of its own sequence (tracking never compares
across videos). Crops of the query's own track within ``window`` seconds are
excluded, so near-duplicate neighbouring frames cannot count as hits.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from reidtrack.retrieval.crops import CropSet


def average_precision(similarity: np.ndarray, positive: np.ndarray) -> float:
    """AP of one query given similarities to the gallery and a positive mask."""
    order = np.argsort(-similarity, kind="stable")
    hits = positive[order]
    if not hits.any():
        return float("nan")
    ranks = np.flatnonzero(hits) + 1
    return float(np.mean(np.arange(1, len(ranks) + 1) / ranks))


def retrieval_scores(
    embed: Callable[[torch.Tensor], torch.Tensor],
    val: CropSet,
    seen_ids: set[int],
    window: float = 1.0,
    query_every: int = 5,
    batch: int = 256,
    device: str = "cuda",
    perturb: Callable[[torch.Tensor, np.ndarray], torch.Tensor] | None = None,
) -> dict[str, float]:
    """mAP and Rank-1 (%) over queries from unseen people; ``embed`` maps uint8 crops to features.

    ``perturb`` changes the query crops only, as ``perturb(crops, rows)`` on uint8 crops, so
    each query is a person seen under other conditions searched against normal sightings.
    """

    def embed_all(change=None):
        out = []
        with torch.inference_mode():
            for i in range(0, len(val), batch):
                chunk = torch.from_numpy(np.array(val.images[i : i + batch])).to(device)
                if change is not None:
                    chunk = change(chunk, np.arange(i, min(i + batch, len(val))))
                out.append(torch.nn.functional.normalize(embed(chunk).float(), dim=1).cpu())
        return torch.cat(out).numpy()

    feats = embed_all()
    queries = feats if perturb is None else embed_all(perturb)
    ids = val.identities()

    aps, top1 = [], []
    for s in np.unique(val.sequence):
        rows = np.flatnonzero(val.sequence == s)
        order = np.lexsort((val.frame[rows], ids[rows]))
        rows = rows[order]
        unseen = np.array([ids[r] not in seen_ids for r in rows])
        is_query = np.zeros(len(rows), dtype=bool)
        for identity in np.unique(ids[rows][unseen]):
            members = np.flatnonzero(ids[rows] == identity)
            is_query[members[::query_every]] = True
        sims = queries[rows[is_query]] @ feats[rows].T
        for q, sim in zip(np.flatnonzero(is_query), sims):
            same = ids[rows] == ids[rows[q]]
            junk = same & (np.abs(val.seconds[rows] - val.seconds[rows[q]]) <= window)
            keep = ~junk
            positive = same[keep]
            if not positive.any():
                continue
            aps.append(average_precision(sim[keep], positive))
            top1.append(bool(positive[np.argmax(sim[keep])]))
    return {"mAP": 100 * float(np.mean(aps)), "rank1": 100 * float(np.mean(top1)), "queries": len(aps)}
