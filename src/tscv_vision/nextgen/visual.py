"""Utilities for visualising pipelines."""
from __future__ import annotations

from .graph import PipelineGraph


def to_dot(graph: PipelineGraph) -> str:
    """Return a Graphviz dot representation of ``graph``."""

    lines = ["digraph G {"]
    for node in graph.nodes.values():
        lines.append(f"  {node.name} [label='{node.name}']")
        for dep in node.deps:
            lines.append(f"  {dep} -> {node.name}")
    lines.append("}")
    return "\n".join(lines)
