"""CUDA-graph replay of a model's forward pass for small, varying batch sizes.

Small networks at small batch sizes spend most of their time launching kernels.
A CUDA graph records the whole forward pass once and replays it as a single
launch. Graphs need fixed shapes, so inputs are padded up to the next bucket.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphedForward:
    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, int, int],
        buckets: tuple[int, ...] = (8, 16, 32, 64),
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> None:
        self.buckets = tuple(sorted(buckets))
        self.dtype = dtype
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode():
            for b in self.buckets:
                static_in = torch.zeros(b, *input_shape, device=device, dtype=dtype)
                static_in = static_in.contiguous(memory_format=memory_format)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        model(static_in)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    static_out = model(static_in)
                self._graphs[b] = (graph, static_in, static_out)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = len(x)
        largest = self.buckets[-1]
        if n > largest:
            return torch.cat([self(x[i : i + largest]) for i in range(0, n, largest)])
        bucket = next(b for b in self.buckets if b >= n)
        graph, static_in, static_out = self._graphs[bucket]
        static_in[:n].copy_(x)
        graph.replay()
        return static_out[:n].clone()
