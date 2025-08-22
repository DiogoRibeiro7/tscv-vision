from __future__ import annotations

import numpy as np
import pytest

from tscv_vision.nextgen import (
    Node,
    PipelineGraph,
    codegen,
    distribute,
    nas,
    optimize,
    quantum,
    registry,
    visual,
)


def test_plugin_registry_and_graph_execution() -> None:
    registry.plugins.clear()

    def double(x: float) -> float:
        return 2 * x

    registry.register("double", double)
    g = PipelineGraph()
    g.add_node(Node(name="out", op="double", deps=["inp"]))
    result = g.run({"inp": 3.0})
    assert result["out"] == 6.0


def test_codegen_and_visual(tmp_path) -> None:
    registry.plugins.clear()
    g = PipelineGraph()
    g.add_node(Node(name="out", op="builtin_abs", deps=["inp"]))
    registry.register("builtin_abs", abs)
    code = codegen.generate_code(g)
    assert "PipelineGraph" in code
    dot = visual.to_dot(g)
    assert "out" in dot and "inp" in dot


def test_distributed_execution() -> None:
    registry.plugins.clear()
    registry.register("builtin_abs", abs)
    g = PipelineGraph()
    g.add_node(Node(name="out", op="builtin_abs", deps=["inp"]))
    res = distribute.execute_distributed(g, {"inp": -1.0}, workers=2)
    assert res["out"] == 1.0


def test_quantum_encoder_guard() -> None:
    with pytest.raises(ImportError):
        quantum.quantum_fourier_encoder(np.arange(8.0))


def test_self_optimizer_and_nas() -> None:
    opt = optimize.SelfOptimizer()
    chosen = opt.choose(lambda: 1, lambda: 2)
    assert callable(chosen)
    best = nas.neural_architecture_search(["a", "b"])
    assert best in {"a", "b"}
