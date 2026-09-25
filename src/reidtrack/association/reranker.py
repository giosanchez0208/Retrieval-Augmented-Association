"""Learned reranker: same-person probability for each (bank entry, detection) pair."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from reidtrack.association.features import NAMES


class PairwiseReranker(nn.Module):
    """A small MLP over the pair cues of ``reidtrack.association.features`` (rung 2 of
    the ablation ladder: learned, but each pair is scored without seeing the others)."""

    def __init__(self, names: tuple[str, ...] = NAMES, hidden: int = 64, dropped: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._set_names(names, dropped)
        self.net = nn.Sequential(
            nn.Linear(len(names), hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def _set_names(self, names: tuple[str, ...], dropped: tuple[str, ...]) -> None:
        """``names`` are the cues the model reads: all of NAMES, or its first few for a model
        trained before later cues were added. ``dropped`` cues are zeroed at training and
        inference, e.g. raw similarities, so the model can only use scale-free ranks."""
        if tuple(NAMES[: len(names)]) != tuple(names):
            raise ValueError("cues must be the first entries of NAMES; retrain the model")
        unknown = set(dropped) - set(names)
        if unknown:
            raise ValueError(f"unknown cues {sorted(unknown)}")
        self.names, self.dropped = tuple(names), tuple(dropped)
        self.register_buffer("keep", torch.tensor([n not in dropped for n in names], dtype=torch.float32))

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        """Logits, shape (..., 1) -> (...)."""
        return self.net(cues[..., : len(self.names)] * self.keep).squeeze(-1)

    def __call__(self, cues: np.ndarray | torch.Tensor, *args, **kwargs):  # type: ignore[override]
        """NumPy cues -> probabilities (used by the tracker); tensors -> logits as usual."""
        if isinstance(cues, np.ndarray):
            if cues.size == 0:
                return np.zeros(cues.shape[:-1], dtype=np.float32)
            with torch.inference_mode():
                return torch.sigmoid(self.forward(torch.from_numpy(cues))).numpy()
        return super().__call__(cues, *args, **kwargs)

    def save(self, path: str | Path, **extra) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": type(self).__name__, "cues": list(self.names), "dropped": list(self.dropped),
                    "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str | Path) -> nn.Module:
        return load_reranker(path)


class AxialBlock(nn.Module):
    """Attention along each row (one entry, all detections), then each column (one
    detection, all entries), then a feed-forward layer; all with residuals."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.row = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.col = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))

    def forward(self, x: torch.Tensor, entry_pad: torch.Tensor, det_pad: torch.Tensor) -> torch.Tensor:
        """x (B, N, M, dim); ``entry_pad`` (B, N) and ``det_pad`` (B, M) mark padding."""
        b, n, m, d = x.shape
        h = self.norms[0](x).reshape(b * n, m, d)
        mask = det_pad[:, None, :].expand(b, n, m).reshape(b * n, m)
        x = x + self.row(h, h, h, key_padding_mask=mask, need_weights=False)[0].reshape(b, n, m, d)
        h = self.norms[1](x).transpose(1, 2).reshape(b * m, n, d)
        mask = entry_pad[:, None, :].expand(b, m, n).reshape(b * m, n)
        x = x + self.col(h, h, h, key_padding_mask=mask, need_weights=False)[0].reshape(b, m, n, d).transpose(1, 2)
        return x + self.ffn(self.norms[2](x))


class AxialReranker(PairwiseReranker):
    """Rung 3 of the ablation ladder: pairs see their competitors.

    Every (entry, detection) pair of a frame is embedded from its cues, then attends
    along its row and its column, so the score of a pair accounts for the other
    candidates of the same entry and of the same detection before assignment.
    """

    def __init__(self, names: tuple[str, ...] = NAMES, dim: int = 32, heads: int = 2, layers: int = 2,
                 dropped: tuple[str, ...] = ()) -> None:
        nn.Module.__init__(self)
        self._set_names(names, dropped)
        self.embed = nn.Sequential(nn.Linear(len(names), dim), nn.ReLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(AxialBlock(dim, heads) for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(
        self, cues: torch.Tensor, entry_pad: torch.Tensor | None = None, det_pad: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Cues (N, M, C) for one frame, or (B, N, M, C) padded frames with their padding
        masks -> logits of the same leading shape."""
        single = cues.dim() == 3
        if single:
            cues = cues[None]
        b, n, m, _ = cues.shape
        if entry_pad is None:
            entry_pad = torch.zeros(b, n, dtype=torch.bool, device=cues.device)
            det_pad = torch.zeros(b, m, dtype=torch.bool, device=cues.device)
        x = self.embed(cues[..., : len(self.names)] * self.keep)
        for block in self.blocks:
            x = block(x, entry_pad, det_pad)
        logits = self.head(x).squeeze(-1)
        return logits[0] if single else logits


def pad_frames(frames: list[tuple[torch.Tensor, torch.Tensor]]):
    """Stack frames of (cues (N, M, C), labels (N, M)) with padding.

    Returns cues (B, N, M, C), labels (B, N, M), entry_pad (B, N), det_pad (B, M), valid (B, N, M).
    """
    n = max(c.shape[0] for c, _ in frames)
    m = max(c.shape[1] for c, _ in frames)
    b, k = len(frames), frames[0][0].shape[-1]
    cues = torch.zeros(b, n, m, k)
    labels = torch.zeros(b, n, m)
    entry_pad = torch.ones(b, n, dtype=torch.bool)
    det_pad = torch.ones(b, m, dtype=torch.bool)
    for i, (c, y) in enumerate(frames):
        cues[i, : c.shape[0], : c.shape[1]] = c
        labels[i, : y.shape[0], : y.shape[1]] = y
        entry_pad[i, : c.shape[0]] = False
        det_pad[i, : c.shape[1]] = False
    valid = ~entry_pad[:, :, None] & ~det_pad[:, None, :]
    return cues, labels, entry_pad, det_pad, valid


def load_reranker(path: str | Path) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    kinds = {"PairwiseReranker": PairwiseReranker, "AxialReranker": AxialReranker}
    try:
        model = kinds[state.get("kind", "PairwiseReranker")](tuple(state["cues"]), dropped=tuple(state.get("dropped", ())))
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from None
    model.load_state_dict(state["state_dict"])
    return model.eval()
