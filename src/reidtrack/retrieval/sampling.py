"""Batches of P people x K crops for metric learning, loaded in a background thread."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

import numpy as np
import torch

from reidtrack.retrieval.crops import CropSet


class SameSceneSampler:
    """Yields index batches of ``people`` identities x ``crops`` crops each.

    All identities in a batch come from one sequence, so negatives share the
    scene, lighting and background, the confusions a tracker faces. Each
    identity's crops are drawn from ``crops`` separate stretches of its track, so
    positives span time instead of being near-duplicate neighbouring frames.
    """

    def __init__(self, data: CropSet, people: int = 16, crops: int = 4, seed: int = 0) -> None:
        self.people, self.crops = people, crops
        self.rng = np.random.default_rng(seed)
        ids = data.identities()
        order = np.lexsort((data.frame, ids))
        bounds = np.flatnonzero(np.diff(ids[order])) + 1
        self.tracks = [t for t in np.split(order, bounds) if len(t) >= 2]
        seq_of = np.array([data.sequence[t[0]] for t in self.tracks])
        self.by_sequence = [np.flatnonzero(seq_of == s) for s in np.unique(seq_of)]
        self.by_sequence = [t for t in self.by_sequence if len(t) >= 2]
        weights = np.array([len(t) for t in self.by_sequence], dtype=float)
        self.sequence_weights = weights / weights.sum()
        self.batches_per_epoch = len(data) // (people * crops)

    def batch(self) -> np.ndarray:
        s = self.rng.choice(len(self.by_sequence), p=self.sequence_weights)
        pool = self.by_sequence[s]
        people = self.rng.choice(pool, size=min(self.people, len(pool)), replace=False)
        out = []
        for p in people:
            track = self.tracks[p]
            stretches = np.array_split(np.arange(len(track)), self.crops)
            out += [track[self.rng.choice(part)] if len(part) else track[self.rng.integers(len(track))] for part in stretches]
        return np.array(out)

    def take(self, batches: int) -> Iterator[np.ndarray]:
        for _ in range(batches):
            yield self.batch()

    def __iter__(self) -> Iterator[np.ndarray]:
        return self.take(self.batches_per_epoch)


class Prefetcher:
    """Reads (images, labels) batches from the memory-mapped crops in a thread.

    ``batches`` defaults to one epoch; a resumed run asks for what is left of one.
    """

    def __init__(
        self, data: CropSet, sampler: SameSceneSampler, labels: np.ndarray, depth: int = 4, batches: int | None = None
    ) -> None:
        self.data, self.sampler, self.labels = data, sampler, labels
        self.batches = sampler.batches_per_epoch if batches is None else batches
        self.pin = torch.cuda.is_available()
        self.queue: queue.Queue = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._fill, daemon=True)
        self.thread.start()

    def _fill(self) -> None:
        for idx in self.sampler.take(self.batches):
            order = np.argsort(idx)  # sorted reads are friendlier to the memory map
            images = np.empty((len(idx), *self.data.images.shape[1:]), dtype=np.uint8)
            images[order] = self.data.images[idx[order]]
            images_t = torch.from_numpy(images)
            self.queue.put((images_t.pin_memory() if self.pin else images_t, torch.from_numpy(self.labels[idx])))
        self.queue.put(None)

    def __iter__(self):
        while (item := self.queue.get()) is not None:
            yield item
