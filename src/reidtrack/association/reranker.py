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

    def __init__(self, cues: int = len(NAMES), hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cues, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        """Logits, shape (..., 1) -> (...)."""
        return self.net(cues).squeeze(-1)

    @torch.inference_mode()
    def __call__(self, cues: np.ndarray | torch.Tensor):  # type: ignore[override]
        if isinstance(cues, np.ndarray):
            if cues.size == 0:
                return np.zeros(cues.shape[:-1], dtype=np.float32)
            return torch.sigmoid(self.forward(torch.from_numpy(cues))).numpy()
        return super().__call__(cues)

    def save(self, path: str | Path, **extra) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": type(self).__name__, "cues": list(NAMES), "state_dict": self.state_dict(), **extra}, path)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, M, dim)
        h = self.norms[0](x)
        x = x + self.row(h, h, h, need_weights=False)[0]
        h = self.norms[1](x).transpose(0, 1)
        x = x + self.col(h, h, h, need_weights=False)[0].transpose(0, 1)
        return x + self.ffn(self.norms[2](x))


class AxialReranker(PairwiseReranker):
    """Rung 3 of the ablation ladder: pairs see their competitors.

    Every (entry, detection) pair of a frame is embedded from its cues, then attends
    along its row and its column, so the score of a pair accounts for the other
    candidates of the same entry and of the same detection before assignment.
    """

    def __init__(self, cues: int = len(NAMES), dim: int = 32, heads: int = 2, layers: int = 2) -> None:
        nn.Module.__init__(self)
        self.embed = nn.Sequential(nn.Linear(cues, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(AxialBlock(dim, heads) for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        """One frame of cues (N, M, C) -> logits (N, M)."""
        x = self.embed(cues)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


def load_reranker(path: str | Path) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(state["cues"]) != NAMES:
        raise ValueError(f"{path} was trained on different cues; retrain it")
    kinds = {"PairwiseReranker": PairwiseReranker, "AxialReranker": AxialReranker}
    model = kinds[state.get("kind", "PairwiseReranker")](len(NAMES))
    model.load_state_dict(state["state_dict"])
    return model.eval()
