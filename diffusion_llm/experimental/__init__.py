"""
AAM Diffusion LLM — Experimental Modules

⚠️  WARNING: These modules are RESEARCH-GRADE and NOT production-ready.
They represent advanced features that were added before baseline training
was established. They are kept for future reference but should NOT be
used until the core architecture has been validated with real training data.

Philosophy (from architecture audit):
  "Buktikan pikiran dulu, baru latih tubuh."
  (Prove the mind first, then train the body.)

  The core pipeline (Layer 0 → RSVS → Layer 2 → Layer 3) has been
  proven end-to-end. The Diffusion LLM "body" needs:
  1. Supervised training with real graph→narrative pairs FIRST
  2. Basic output quality validation
  3. ONLY THEN can these advanced features be incrementally added

Premature features moved here:
  - Evoformer (AlphaFold2-style bidirectional feedback)
  - DualMemory (working + long-term memory system)
  - MCTS (Monte Carlo Tree Search for reasoning)
  - DAPO (DeepSeek-R1-style RL alignment)
  - GRPO (Group Relative Policy Optimization)
  - FlowMatching (alternative velocity-based sampling)
  - AnchoredDecoder (2-3 step refinement)
  - ThinkingToggle (adaptive compute depth)
  - Matryoshka (elastic inference at multiple sizes)
  - SpeculativeDecoder + MirrorSpeculative (speculative decoding)
  - Quantization (BitLinear/FP8Linear)
  - JEPA (Joint-Embedding Predictive Architecture)
  - Curriculum (curriculum learning scheduler)

These are NOT deleted — they are preserved for when the core
architecture is validated and there's real training data to
justify their addition.
"""

# Lazy imports — only load when explicitly requested
_EXPERIMENTAL_MODULES = {
    "evoformer": ".evoformer",
    "dual_memory": ".dual_memory",
    "anchored_decoder": ".anchored_decoder",
    "flow_matching": ".flow_matching",
    "thinking_toggle": ".thinking_toggle",
    "mcts": ".mcts",
    "matryoshka": ".matryoshka",
    "mirror_speculative": ".mirror_speculative",
    "speculative_decoder": ".speculative_decoder",
    "quantization": ".quantization",
    "grpo": ".grpo",
    "dapo": ".dapo",
    "curriculum": ".curriculum",
    "llm_jepa": ".llm_jepa",
}


def __getattr__(name: str):
    """Lazy import for experimental modules."""
    if name in _EXPERIMENTAL_MODULES:
        import importlib
        module = importlib.import_module(_EXPERIMENTAL_MODULES[name], __package__)
        return module
    raise AttributeError(f"module 'diffusion_llm.experimental' has no attribute {name!r}")
