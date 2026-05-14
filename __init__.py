"""
AAM — Aphantasic Abstraction Model
===================================

"Bukan AI yang menyimpan foto. AI yang memahami relasi."

Architecture:
  layer0/    Perceptual Front-End    — raw input → structured tuples
  layer1/    Abstraction Engine      — tuples → knowledge graph (RSVS Rust core)
  layer2/    Cognitive Runtime       — context, situation, prediction, pattern
  layer3/    Deductive Reasoning     — policy, coder, traceable output
  pipeline.py                        — AamPipeline (wires all layers)

Inspired by Aphantasia: the cognitive condition where no visual imagery
is stored — only relational structure. This is how AAM remembers.
"""

__version__ = "8.6.0"
__name_full__ = "Aphantasic Abstraction Model"
__name_short__ = "AAM"

try:
    from .pipeline import AamPipeline, AamResponse
except ImportError:
    # Pipeline requires full layer2/layer3 dependencies
    # which may not be available in all environments
    pass
