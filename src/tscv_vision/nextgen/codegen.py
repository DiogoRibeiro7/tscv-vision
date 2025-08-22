"""Automatic code generation from pipeline graphs."""
from __future__ import annotations

from .graph import PipelineGraph


def generate_code(graph: PipelineGraph) -> str:
    """Return Python code that recreates ``graph``.

    The generated code assumes all required plugins are registered under the
    same names.
    """

    lines = ["from tscv_vision.nextgen import PipelineGraph, Node", "g = PipelineGraph()"]
    for node in graph.topological():
        op_name = node.op if isinstance(node.op, str) else node.op.__name__
        deps = list(node.deps)
        lines.append(
            f"g.add_node(Node(name='{node.name}', op='{op_name}', deps={deps}))"
        )
    lines.append("result = g.run(inputs)")
    return "\n".join(lines)
