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

Architecture:
  Input: Graph conditioning (evidence nodes, compositions, confidence, anomalies)
  Process: Iterative denoising from noise
  Output: Natural language narrative grounded in graph structure

Analogi: Jin Soun (graph) + tubuhnya (this model).
Tubuhnya third-rate, tapi karena KHUSUS dilatih untuk
mengeksekusi perintah dari graph-nya sendiri, outputnya
lebih terarah daripada LLM umum yang "tidak kenal" graph.
"""

__version__ = "2.1.0"
__author__ = "AAM Team"

from diffusion_llm.config.model_config import AamDiffusionConfig, get_default_config
from diffusion_llm.model.noise_scheduler import NoiseScheduler
from diffusion_llm.model.graph_encoder import GraphConditioningEncoder
from diffusion_llm.model.diffusion_transformer import DiffusionTransformer
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer
from diffusion_llm.inference.generator import AamGenerator
from diffusion_llm.training.trainer import AamTrainer
from diffusion_llm.training.dataset import GraphNarrativeDataset
from diffusion_llm.data.synthetic_generator import SyntheticDataGenerator

# v2.0 modules (from Losion upgrade)
from diffusion_llm.model.anchored_decoder import AnchoredDiffusionDecoder, ContinuousOutputHead
from diffusion_llm.model.flow_matching import FlowMatchingDecoder
from diffusion_llm.model.evoformer import EvoformerManager, RouterExpertCoevolve
from diffusion_llm.model.dual_memory import DualMemorySystem
from diffusion_llm.model.mcts import MCTSReasoner
from diffusion_llm.model.thinking_toggle import ThinkingToggle, ThinkingMode
from diffusion_llm.model.matryoshka import MatryoshkaLayer, ElasticExtractor
from diffusion_llm.model.rope import RotaryPositionEncoding
from diffusion_llm.model.speculative_decoder import SpeculativeDecoder
from diffusion_llm.model.mirror_speculative import MirrorSpeculativeDecoder, MirrorSpeculativeConfig
from diffusion_llm.model.quantization import BitLinear, FP8Linear
from diffusion_llm.training.grpo import GRPOTrainer
from diffusion_llm.training.dapo import DAPOTrainer
from diffusion_llm.training.curriculum import CurriculumScheduler
from diffusion_llm.training.llm_jepa import JEPAPredictor, JEPAConfig, JEPATrainer

__all__ = [
    # Core
    "AamDiffusionConfig",
    "get_default_config",
    "NoiseScheduler",
    "GraphConditioningEncoder",
    "DiffusionTransformer",
    "AamDiffusionModel",
    "AamTokenizer",
    "AamGenerator",
    "AamTrainer",
    "GraphNarrativeDataset",
    "SyntheticDataGenerator",
    # v2.0 — Losion Upgrade
    "AnchoredDiffusionDecoder",
    "ContinuousOutputHead",
    "FlowMatchingDecoder",
    "EvoformerManager",
    "RouterExpertCoevolve",
    "DualMemorySystem",
    "MCTSReasoner",
    "ThinkingToggle",
    "ThinkingMode",
    "MatryoshkaLayer",
    "ElasticExtractor",
    "RotaryPositionEncoding",
    "SpeculativeDecoder",
    "BitLinear",
    "FP8Linear",
    "GRPOTrainer",
    "DAPOTrainer",
    "CurriculumScheduler",
    # v2.1 — Mirror Speculative & JEPA
    "MirrorSpeculativeDecoder",
    "MirrorSpeculativeConfig",
    "JEPAPredictor",
    "JEPAConfig",
    "JEPATrainer",
]
