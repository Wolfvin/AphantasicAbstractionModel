# @WHO:   self-ai/src/unconscious/__init__.py
# @WHAT:  Unconscious experience injection — activation steering for Qwen3
# @PART:  self-ai/unconscious

"""Unconscious experience injection module.

SELF's understanding graph currently influences answers by injecting
retrieved understandings into the text prompt (conscious path).
This module provides an UNCONSCIOUS alternative: inject experience
vectors directly into Qwen3's hidden states during forward pass,
so the model "feels" the experience without seeing it in the prompt.
"""

from unconscious.injector import UnconsciousInjector

__all__ = ['UnconsciousInjector']
