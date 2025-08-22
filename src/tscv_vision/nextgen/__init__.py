"""Next-generation architecture components for tscv-vision 2.0.

This package groups experimental features such as a plugin registry,
graph-based pipelines and code generation. These utilities are optional and
intended for advanced users.
"""
from .graph import Node, PipelineGraph
from .plugins import PluginRegistry, registry

__all__ = ["PluginRegistry", "registry", "Node", "PipelineGraph"]
