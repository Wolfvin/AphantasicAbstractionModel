# @WHO:   self-ai/src/introspection/__init__.py
# @WHAT:  Introspection module — SELF explains WHY it answered the way it did
# @PART:  self-ai/introspection

"""Introspection module.

When SELF answers via the unconscious path (activation steering),
the user might ask "why did you answer that way?". Introspector
traces back to the UnderstandingNodes that were injected and
generates a natural-language explanation.

This is NOT template-based — Qwen3 generates the explanation,
using the node content as context. The model articulates its
own reasoning, which is the whole point: SELF can introspect.
"""

from introspection.introspector import Introspector

__all__ = ['Introspector']
