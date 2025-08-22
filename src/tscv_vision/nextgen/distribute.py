"""Distributed execution helpers."""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

from .graph import PipelineGraph


def execute_distributed(
    graph: PipelineGraph, inputs: dict[str, Any], workers: int = 2
) -> dict[str, Any]:
    """Execute ``graph`` using a process pool.

    For small graphs this falls back to the sequential ``PipelineGraph.run``.
    """

    if workers < 2:
        return graph.run(inputs)

    order = graph.topological()
    data = dict(inputs)
    with mp.Pool(processes=workers) as pool:
        for node in order:
            args = [data[d] for d in node.deps]
            data[node.name] = pool.apply(node.resolve(), args)
    return data
