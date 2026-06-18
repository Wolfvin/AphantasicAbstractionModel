"""
ENGRAM COMPLEX: Connected memory network (graph).

Biologis: Memory is distributed across hippocampus + neocortex as graph.
AI: Container for episomes (nodes) and semesomes (edges).

IMPORTANT: This WRAPS (does NOT replace) AGNNGraph from self-ai/src/agnn/graph.py.
We delegate storage and traversal to the existing AGNNGraph implementation.

The AGNNGraph import is deferred to __init__ so that this module can be
imported (and other engrams submodules tested) even when self-ai/ is not
present on sys.path. The wrap contract is enforced at construction time.
"""

import os
import sys


def _resolve_agnn_graph():
    """Import AGNNGraph from self-ai/src/agnn/graph.py at call time.

    Adds self-ai/src to sys.path (idempotent) and returns the AGNNGraph class.

    Raises:
        ImportError: if self-ai/src/agnn/graph.py is not importable.
    """
    self_ai_src = os.path.join(os.path.dirname(__file__), "..", "..", "self-ai", "src")
    self_ai_src = os.path.abspath(self_ai_src)
    if self_ai_src not in sys.path:
        sys.path.insert(0, self_ai_src)
    from agnn.graph import AGNNGraph  # reuse existing implementation
    return AGNNGraph


class EngramComplex:
    """Connected memory network - wraps AGNNGraph from self-ai/src/agnn/.

    This class delegates storage and traversal to the existing AGNNGraph
    implementation. It does NOT reimplement graph logic; it adds the
    neuroanatomical abstraction layer (episome/semesome naming, engram audit).

    Attributes:
        _graph: The wrapped AGNNGraph instance.
    """

    def __init__(self):
        """Wrap a fresh AGNNGraph instance.

        Raises:
            ImportError: if self-ai/src/agnn/graph.py cannot be imported.
        """
        AGNNGraph = _resolve_agnn_graph()
        # Delegate to existing AGNNGraph - do NOT reimplement.
        self._graph = AGNNGraph()
