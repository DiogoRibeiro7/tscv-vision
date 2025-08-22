"""Graph-based computation engine for feature pipelines."""
from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .plugins import registry


@dataclass
class Node:
    """A node in the computation graph."""

    name: str
    op: Callable[[Any], Any] | str
    deps: Sequence[str]

    def resolve(self) -> Callable[[Any], Any]:
        """Return the callable implementing this node."""

        return registry.get(self.op) if isinstance(self.op, str) else self.op


class PipelineGraph:
    """Directed acyclic graph of computation nodes."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name!r} already exists")
        self.nodes[node.name] = node

    def topological(self) -> list[Node]:
        """Topologically sorted list of nodes."""

        indegree: dict[str, int] = {n: 0 for n in self.nodes}
        for node in self.nodes.values():
            for dep in node.deps:
                indegree.setdefault(dep, 0)
                indegree[dep] += 1
        queue = deque([self.nodes[n] for n, d in indegree.items() if d == 0])
        order: list[Node] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.nodes.values():
                if node.name in child.deps:
                    indegree[child.name] -= 1
                    if indegree[child.name] == 0:
                        queue.append(child)
        if len(order) != len(self.nodes):  # pragma: no cover - defensive
            raise ValueError("Graph has cycles")
        return order

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the graph given initial inputs."""

        data = dict(inputs)
        for node in self.topological():
            args = [data[d] for d in node.deps]
            data[node.name] = node.resolve()(*args)
        return data
