"""
AAM Diffusion LLM Framework — The Body of Aphantasic Abstraction Model

"AAM = 1 Pikiran + 1 Tubuh" (1 Mind + 1 Body)

Pikiran (Mind) = RSVS Knowledge Graph — structural, relational, perfect memory
Tubuh (Body)  = This Diffusion LLM — generates natural language FROM the graph

This is NOT a general-purpose LLM. This is a SPECIALIZED sentence composer
that takes structured graph data as input and produces coherent, evidence-backed
narrative output. Think of it as a "vocal cord" for the graph — it can only
say what the graph knows, but it says it fluently.

Why Diffusion?
- Diffusion models start from noise and iteratively denoise
- This mirrors how Jin Soun's thoughts form: from vague intuition ->
  clearer pattern -> explicit narrative
- Unlike autoregressive LLMs (GPT), diffusion models can:
  - Be conditioned on structured input (graph)
  - Revise earlier parts during generation (non-sequential)
  - Produce more coherent long-form text from structure

Architecture (Core):
  Input: Graph conditioning (evidence nodes, compositions, confidence, anomalies)
  Process: Iterative denoising from noise
  Output: Natural language narrative grounded in graph structure

Philosophy: "Buktikan pikiran dulu, baru latih tubuh."
(Prove the mind first, then train the body.)

The mind pipeline (Layer 0 → RSVS → Layer 2 → Layer 3) has been proven
end-to-end. The body needs supervised training with real graph→narrative
pairs BEFORE adding advanced features.

Experimental modules are in diffusion_llm.experimental/ — see that
package for research-grade features that are preserved but not yet
validated with real training data.
"""

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

__version__ = "2.2.0"
__author__ = "AAM Team"

# Core architecture — always available
from diffusion_llm.config.model_config import AamDiffusionConfig, get_default_config
from diffusion_llm.model.noise_scheduler import NoiseScheduler
from diffusion_llm.model.graph_encoder import GraphConditioningEncoder
from diffusion_llm.model.diffusion_transformer import DiffusionTransformer
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.model.rope import RotaryPositionEncoding

# Core infrastructure — always available
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer
from diffusion_llm.inference.generator import AamGenerator
from diffusion_llm.training.trainer import AamTrainer
from diffusion_llm.training.dataset import GraphNarrativeDataset
from diffusion_llm.training.losses import DiffusionLoss, compute_loss

__all__ = [
    # Core architecture
    "AamDiffusionConfig",
    "get_default_config",
    "NoiseScheduler",
    "GraphConditioningEncoder",
    "DiffusionTransformer",
    "AamDiffusionModel",
    "RotaryPositionEncoding",
    # Core infrastructure
    "AamTokenizer",
    "AamGenerator",
    "AamTrainer",
    "GraphNarrativeDataset",
    "DiffusionLoss",
    "compute_loss",
]
