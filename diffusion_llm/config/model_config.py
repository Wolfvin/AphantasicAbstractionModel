"""
AAM Diffusion LLM — Model Configuration

Defines all hyperparameters for the diffusion model architecture,
training process, and inference pipeline.

Design Philosophy:
    - Small model (100M-500M params) — specialized, not general
    - Sentence-level tokenization — not subword, because AAM arranges
      sentences, not individual tokens
    - Graph-conditioned — the model MUST receive graph structure as input
    - Non-sequential generation — diffusion, not autoregressive

Analogi: Seperti tubuh Jin Soun, model ini kecil tapi KKHUSUS
dilatih untuk satu tugas: menarasikan dari graph. Tidak perlu
7B params kalau tugasku hanya menyusun kalimat dari data yang
sudah terstruktur.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture hyperparameters for the Diffusion Transformer.

    Target: 100M-500M parameters total.
    Calculation:
        params ≈ d_model^2 * (12 * n_layers) for transformer
        d_model=512, n_layers=8  → ~50M core params
        d_model=768, n_layers=12 → ~170M core params
        d_model=1024, n_layers=12 → ~300M core params
    """

    # --- Core Transformer ---
    d_model: int = 768
    """Hidden dimension of the transformer."""

    n_layers: int = 12
    """Number of transformer blocks."""

    n_heads: int = 12
    """Number of attention heads (d_model must be divisible by n_heads)."""

    d_ff: int = 3072
    """Feed-forward hidden dimension (typically 4x d_model)."""

    dropout: float = 0.1
    """Dropout rate for attention and feed-forward layers."""

    activation: str = "gelu"
    """Activation function: 'gelu' or 'relu'."""

    # --- Sequence ---
    max_seq_len: int = 512
    """Maximum sequence length (in sentence-level tokens)."""

    # --- Vocabulary ---
    vocab_size: int = 32000
    """Vocabulary size for the tokenizer.
    Since we use sentence-level tokens + subword BPE hybrid,
    this includes special tokens + subword units.
    """

    # --- Positional Encoding ---
    pos_encoding_type: str = "rotary"
    """Positional encoding type: 'rotary' (RoPE) or 'learned'."""

    # --- Attention ---
    use_flash_attention: bool = True
    """Whether to use Flash Attention 2 if available."""

    # --- Normalization ---
    norm_type: str = "rmsnorm"
    """Normalization type: 'rmsnorm' or 'layernorm'."""

    norm_eps: float = 1e-6
    """Epsilon for normalization layers."""

    # --- Initialization ---
    init_std: float = 0.02
    """Standard deviation for weight initialization."""

    def estimate_params(self) -> str:
        """Estimate total parameter count."""
        # Embedding: vocab_size * d_model
        embed_params = self.vocab_size * self.d_model
        # Per layer: 4 * d_model^2 (QKV + O) + 2 * d_model * d_ff (FF)
        layer_params = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff
        total = embed_params + self.n_layers * layer_params
        if total >= 1e9:
            return f"{total / 1e9:.1f}B"
        elif total >= 1e6:
            return f"{total / 1e6:.1f}M"
        else:
            return f"{total / 1e3:.1f}K"


@dataclass
class DiffusionConfig:
    """Hyperparameters for the diffusion process.

    The diffusion process works on the latent representation of text:
    1. Forward: Add Gaussian noise to text embeddings over T timesteps
    2. Reverse: Learn to denoise step by step
    3. At inference: Start from pure noise, denoise to coherent text

    This is DIFFERENT from image diffusion because:
    - We operate in a learned latent space (not pixel space)
    - Text has discrete structure (sentences, not pixels)
    - We use a text-specific noise schedule
    """

    # --- Noise Schedule ---
    n_timesteps: int = 1000
    """Total number of diffusion timesteps for training."""

    n_inference_steps: int = 50
    """Number of denoising steps at inference (fewer = faster, less quality)."""

    schedule_type: str = "cosine"
    """Noise schedule type: 'linear', 'cosine', or 'sigmoid'."""

    beta_start: float = 1e-4
    """Starting beta for linear schedule."""

    beta_end: float = 0.02
    """Ending beta for linear schedule."""

    # --- Noise Prediction ---
    prediction_type: str = "epsilon"
    """What the model predicts: 'epsilon' (noise), 'x0' (clean data),
    or 'v' (velocity). Epsilon prediction is most stable for text."""

    # --- Sampling ---
    sampling_method: str = "ddim"
    """Sampling method: 'ddpm' (slow, stochastic) or 'ddim' (fast, deterministic)."""

    eta_ddim: float = 0.0
    """DDIM stochasticity parameter (0 = deterministic, 1 = full stochastic)."""

    # --- Clipping ---
    clip_sample_max: float = 5.0
    """Maximum value for clipped samples during inference."""

    clip_sample_min: float = -5.0
    """Minimum value for clipped samples during inference."""

    # --- Loss ---
    loss_type: str = "mse"
    """Loss function: 'mse' (L2) or 'mae' (L1) or 'huber'."""

    loss_weighting: str = "min_snr"
    """Loss weighting strategy: 'none', 'min_snr', or 'p2'."""

    p2_gamma: float = 1.0
    """P2 weighting gamma (only used if loss_weighting='p2')."""

    p2_k: float = 1.0
    """P2 weighting k (only used if loss_weighting='p2')."""


@dataclass
class GraphEncoderConfig:
    """Configuration for the Graph Conditioning Encoder.

    The graph encoder takes structured graph data (evidence nodes,
    compositions, confidence scores, anomalies, reasoning chains)
    and produces a conditioning vector that guides the diffusion process.

    This is the KEY differentiator from general LLMs:
    the model is conditioned on GRAPH STRUCTURE, not just text prompts.
    """

    # --- Graph Encoder Architecture ---
    d_graph: int = 512
    """Hidden dimension for graph encoding."""

    n_graph_layers: int = 4
    """Number of graph attention layers."""

    n_graph_heads: int = 8
    """Number of attention heads for graph encoding."""

    # --- Input Dimensions ---
    max_evidence_nodes: int = 50
    """Maximum number of evidence nodes to encode."""

    max_compositions: int = 20
    """Maximum number of compositions to encode."""

    max_anomalies: int = 10
    """Maximum number of anomalies to encode."""

    max_reasoning_steps: int = 15
    """Maximum number of reasoning steps to encode."""

    # --- Conditioning Injection ---
    conditioning_method: str = "cross_attention"
    """How to inject graph conditioning into the diffusion model:
    'cross_attention' (separate encoder, cross-attn in transformer)
    'ada_ln' (adaptive layer norm, conditioning modulates scale/shift)
    'concat' (concatenate conditioning to input sequence)
    """

    # --- Confidence Embedding ---
    embed_confidence: bool = True
    """Whether to embed confidence scores as part of the conditioning."""

    # --- Temporal Embedding ---
    embed_temporal: bool = True
    """Whether to embed temporal context (time-based relationships)."""


@dataclass
class TokenizerConfig:
    """Configuration for the AAM Sentence-Level Tokenizer.

    Unlike standard BPE tokenizers that operate at subword level,
    AAM's tokenizer is designed for SENTENCE ARRANGEMENT:
    - Sentences are the primary unit of generation
    - Within sentences, subword BPE handles individual words
    - Special tokens for graph structure (evidence, anomaly, etc.)
    """

    # --- BPE ---
    bpe_vocab_size: int = 28000
    """Subword BPE vocabulary size (within the total vocab_size)."""

    # --- Sentence-Level ---
    max_sentences: int = 32
    """Maximum number of sentences in one generation."""

    sentence_boundary_token: str = "<sent>"
    """Token marking sentence boundaries."""

    # --- Special Tokens ---
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    mask_token: str = "<mask>"
    noise_token: str = "<noise>"

    # --- Graph-Structure Tokens ---
    evidence_token: str = "<evidence>"
    anomaly_token: str = "<anomaly>"
    confidence_token: str = "<confidence>"
    reasoning_token: str = "<reasoning>"
    composition_token: str = "<composition>"
    temporal_token: str = "<temporal>"

    # --- Training ---
    min_frequency: int = 2
    """Minimum frequency for BPE merge operations."""

    dropout_rate: float = 0.0
    """BPE dropout rate (0 = no dropout, regularization during training)."""


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""

    # --- Optimizer ---
    learning_rate: float = 1e-4
    """Peak learning rate."""

    weight_decay: float = 0.01
    """Weight decay for AdamW."""

    adam_beta1: float = 0.9
    """Adam beta1."""

    adam_beta2: float = 0.999
    """Adam beta2."""

    adam_eps: float = 1e-8
    """Adam epsilon."""

    # --- Learning Rate Schedule ---
    lr_schedule: str = "cosine"
    """LR schedule: 'cosine', 'linear', or 'constant'."""

    warmup_steps: int = 2000
    """Number of warmup steps."""

    # --- Training ---
    batch_size: int = 32
    """Training batch size (per GPU)."""

    gradient_accumulation_steps: int = 4
    """Gradient accumulation steps (effective batch = batch_size * this)."""

    max_steps: int = 500000
    """Maximum training steps."""

    max_epochs: int = 100
    """Maximum training epochs."""

    # --- Regularization ---
    dropout: float = 0.1
    """Training dropout rate."""

    grad_clip_norm: float = 1.0
    """Gradient clipping max norm."""

    # --- Mixed Precision ---
    use_amp: bool = True
    """Whether to use Automatic Mixed Precision (fp16/bf16)."""

    amp_dtype: str = "bf16"
    """AMP data type: 'fp16' or 'bf16'."""

    # --- Checkpointing ---
    save_every_steps: int = 5000
    """Save checkpoint every N steps."""

    eval_every_steps: int = 1000
    """Evaluate every N steps."""

    keep_last_n_checkpoints: int = 3
    """Keep only the last N checkpoints."""

    # --- EMA ---
    use_ema: bool = True
    """Whether to use Exponential Moving Average for inference weights."""

    ema_decay: float = 0.9999
    """EMA decay rate."""

    # --- Data ---
    train_data_path: str = ""
    """Path to training data (JSONL format)."""

    val_data_path: str = ""
    """Path to validation data (JSONL format)."""

    num_workers: int = 4
    """Number of data loading workers."""

    # --- Logging ---
    log_every_steps: int = 100
    """Log training metrics every N steps."""

    wandb_project: str = "aam-diffusion-llm"
    """Weights & Biases project name."""

    wandb_run_name: str = ""
    """Weights & Biases run name (auto-generated if empty)."""


@dataclass
class InferenceConfig:
    """Inference-time configuration."""

    n_steps: int = 50
    """Number of denoising steps (more = better quality, slower)."""

    temperature: float = 1.0
    """Sampling temperature (1.0 = standard, <1 = more deterministic)."""

    top_k: int = 50
    """Top-k sampling for token decoding."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    repetition_penalty: float = 1.2
    """Penalty for repeating tokens."""

    max_output_sentences: int = 16
    """Maximum number of sentences in output."""

    language: str = "id"
    """Output language: 'id' (Indonesian) or 'en' (English)."""


# ---------------------------------------------------------------------------
# v2.0 Upgrade — New Module Configurations (from Losion)
# ---------------------------------------------------------------------------

@dataclass
class AnchoredDecoderConfig:
    """Configuration for Anchored Diffusion Decoder."""

    d_model: int = 768
    d_vocab: int = 32000
    n_refine_steps: int = 3
    d_refine: int = 512
    use_evoformer_feedback: bool = True
    n_feedback_iterations: int = 2
    disambiguation_heads: int = 8


@dataclass
class FlowMatchingConfig:
    """Configuration for Flow Matching Decoder."""

    d_model: int = 768
    d_vocab: int = 32000
    num_steps: int = 3


@dataclass
class EvoformerConfig:
    """Configuration for Evoformer Feedback System."""

    d_model: int = 768
    n_recycling_steps: int = 3
    dropout: float = 0.0
    use_layer_recycling: bool = True
    use_token_recycling: bool = True
    use_decoder_feedback: bool = True
    use_prediction_recycling: bool = True
    min_recycling_improvement: float = 1e-4


@dataclass
class DualMemoryConfig:
    """Configuration for Dual Memory System."""

    d_model: int = 768
    working_memory_size: int = 512
    long_term_memory_dim: int = 256
    consolidation_method: str = "attention"
    retrieval_method: str = "attention"
    n_retrieval_heads: int = 4
    dropout: float = 0.0


@dataclass
class MCTSConfig:
    """Configuration for MCTS Reasoning Engine."""

    num_simulations: int = 64
    c_puct: float = 1.5
    temperature: float = 1.0
    max_depth: int = 10
    use_value_network: bool = True
    max_children: int = 8


@dataclass
class ThinkingToggleConfig:
    """Configuration for Thinking Toggle."""

    d_model: int = 768
    threshold: float = 0.5


@dataclass
class MatryoshkaConfig:
    """Configuration for Matryoshka Elastic Inference."""

    d_model: int = 768
    d_ff: int = 3072
    granularity_factors: list = None  # will use default_factory in __post_init__
    matryoshka_loss_weight: float = 0.1
    use_adaptive: bool = True

    def __post_init__(self):
        if self.granularity_factors is None:
            self.granularity_factors = [0.25, 0.5, 0.75, 1.0]


@dataclass
class AamDiffusionConfig:
    """Master configuration for the AAM Diffusion LLM.

    Combines all sub-configurations into a single object.
    This is the entry point for configuring the entire framework.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    graph_encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # --- v2.0 Upgrades from Losion ---
    anchored_decoder: AnchoredDecoderConfig = field(default_factory=AnchoredDecoderConfig)
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
    evoformer: EvoformerConfig = field(default_factory=EvoformerConfig)
    dual_memory: DualMemoryConfig = field(default_factory=DualMemoryConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    thinking_toggle: ThinkingToggleConfig = field(default_factory=ThinkingToggleConfig)
    matryoshka: MatryoshkaConfig = field(default_factory=MatryoshkaConfig)

    # --- v2.0 Feature Flags ---
    use_anchored_decoder: bool = True
    use_flow_matching: bool = True
    use_evoformer: bool = True
    use_dual_memory: bool = True
    use_mcts: bool = False  # Future — needs custom state representation
    use_thinking_toggle: bool = True
    use_matryoshka: bool = True
    use_swiglu_ffn: bool = True  # Replace GELU with SwiGLU

    # --- Meta ---
    model_name: str = "aam-diffusion-v2.0"
    """Model name for saving/loading."""

    output_dir: str = "./output"
    """Base output directory."""

    seed: int = 42
    """Random seed for reproducibility."""

    # --- AAM Philosophy ---
    aam_mind_source: str = "rsvs_graph"
    """Source of the 'mind' that conditions this 'body'.
    Always 'rsvs_graph' for AAM — the model CANNOT generate
    information not present in the graph conditioning."""

    aam_body_type: str = "specialized_diffusion"
    """Type of the 'body'. Always 'specialized_diffusion' for AAM.
    This is NOT a general LLM — it only arranges sentences
    based on graph-structured evidence."""

    def to_dict(self) -> dict:
        """Serialize config to dictionary."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> AamDiffusionConfig:
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            diffusion=DiffusionConfig(**data.get("diffusion", {})),
            graph_encoder=GraphEncoderConfig(**data.get("graph_encoder", {})),
            tokenizer=TokenizerConfig(**data.get("tokenizer", {})),
            training=TrainingConfig(**data.get("training", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            # v2.0 sub-configs
            anchored_decoder=AnchoredDecoderConfig(**data.get("anchored_decoder", {})),
            flow_matching=FlowMatchingConfig(**data.get("flow_matching", {})),
            evoformer=EvoformerConfig(**data.get("evoformer", {})),
            dual_memory=DualMemoryConfig(**data.get("dual_memory", {})),
            mcts=MCTSConfig(**data.get("mcts", {})),
            thinking_toggle=ThinkingToggleConfig(**data.get("thinking_toggle", {})),
            matryoshka=MatryoshkaConfig(**data.get("matryoshka", {})),
            # v2.0 feature flags
            use_anchored_decoder=data.get("use_anchored_decoder", True),
            use_flow_matching=data.get("use_flow_matching", True),
            use_evoformer=data.get("use_evoformer", True),
            use_dual_memory=data.get("use_dual_memory", True),
            use_mcts=data.get("use_mcts", False),
            use_thinking_toggle=data.get("use_thinking_toggle", True),
            use_matryoshka=data.get("use_matryoshka", True),
            use_swiglu_ffn=data.get("use_swiglu_ffn", True),
            # Meta
            model_name=data.get("model_name", "aam-diffusion-v2.0"),
            output_dir=data.get("output_dir", "./output"),
            seed=data.get("seed", 42),
            aam_mind_source=data.get("aam_mind_source", "rsvs_graph"),
            aam_body_type=data.get("aam_body_type", "specialized_diffusion"),
        )

    def summary(self) -> str:
        """Print a summary of the configuration."""
        lines = [
            "=" * 60,
            f"  AAM Diffusion LLM Configuration: {self.model_name}",
            "=" * 60,
            "",
            f"  Model Architecture:",
            f"    d_model={self.model.d_model}, n_layers={self.model.n_layers}, "
            f"n_heads={self.model.n_heads}",
            f"    d_ff={self.model.d_ff}, vocab_size={self.model.vocab_size}",
            f"    max_seq_len={self.model.max_seq_len}",
            f"    Estimated params: {self.model.estimate_params()}",
            "",
            f"  Diffusion Process:",
            f"    Timesteps (train)={self.diffusion.n_timesteps}",
            f"    Timesteps (inference)={self.diffusion.n_inference_steps}",
            f"    Schedule={self.diffusion.schedule_type}",
            f"    Prediction={self.diffusion.prediction_type}",
            f"    Sampling={self.diffusion.sampling_method}",
            "",
            f"  Graph Encoder:",
            f"    d_graph={self.graph_encoder.d_graph}",
            f"    n_layers={self.graph_encoder.n_graph_layers}",
            f"    Conditioning={self.graph_encoder.conditioning_method}",
            f"    Max evidence nodes={self.graph_encoder.max_evidence_nodes}",
            "",
            f"  Training:",
            f"    LR={self.training.learning_rate}",
            f"    Batch={self.training.batch_size} x {self.training.gradient_accumulation_steps} accum",
            f"    Max steps={self.training.max_steps}",
            f"    AMP={self.training.use_amp} ({self.training.amp_dtype})",
            "",
            f"  v2.0 Modules (Losion Upgrade):",
            f"    Anchored Decoder: {self.use_anchored_decoder} "
            f"(n_refine={self.anchored_decoder.n_refine_steps})",
            f"    Flow Matching:    {self.use_flow_matching} "
            f"(num_steps={self.flow_matching.num_steps})",
            f"    Evoformer:        {self.use_evoformer} "
            f"(n_recycle={self.evoformer.n_recycling_steps})",
            f"    Dual Memory:      {self.use_dual_memory} "
            f"(working={self.dual_memory.working_memory_size})",
            f"    MCTS:             {self.use_mcts} "
            f"(simulations={self.mcts.num_simulations})",
            f"    Thinking Toggle:  {self.use_thinking_toggle} "
            f"(threshold={self.thinking_toggle.threshold})",
            f"    Matryoshka:       {self.use_matryoshka} "
            f"(factors={self.matryoshka.granularity_factors})",
            f"    SwiGLU FFN:       {self.use_swiglu_ffn}",
            "",
            f"  AAM Philosophy:",
            f"    Mind = {self.aam_mind_source} (RSVS Knowledge Graph)",
            f"    Body = {self.aam_body_type} (This Model)",
            f"    Identity = 1 Mind + 1 Body (NOT rented LLM)",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)


def get_default_config(
    model_size: str = "base",
) -> AamDiffusionConfig:
    """Get a default configuration for different model sizes.

    Args:
        model_size: One of 'tiny', 'small', 'base', 'medium'.
            - tiny:   ~25M params  (for quick testing)
            - small:  ~70M params  (for development)
            - base:   ~170M params (recommended for training)
            - medium: ~300M params (for final training)

    Returns:
        AamDiffusionConfig with appropriate settings.
    """
    configs = {
        "tiny": AamDiffusionConfig(
            model=ModelConfig(
                d_model=256,
                n_layers=4,
                n_heads=4,
                d_ff=1024,
                vocab_size=16000,
                max_seq_len=256,
            ),
            graph_encoder=GraphEncoderConfig(
                d_graph=256,
                n_graph_layers=2,
                n_graph_heads=4,
            ),
            diffusion=DiffusionConfig(
                n_timesteps=500,
                n_inference_steps=20,
            ),
            training=TrainingConfig(
                batch_size=16,
                learning_rate=3e-4,
                warmup_steps=500,
                max_steps=100000,
            ),
            model_name="aam-diffusion-tiny",
        ),
        "small": AamDiffusionConfig(
            model=ModelConfig(
                d_model=512,
                n_layers=8,
                n_heads=8,
                d_ff=2048,
                vocab_size=24000,
                max_seq_len=384,
            ),
            graph_encoder=GraphEncoderConfig(
                d_graph=384,
                n_graph_layers=4,
                n_graph_heads=8,
            ),
            diffusion=DiffusionConfig(
                n_timesteps=1000,
                n_inference_steps=30,
            ),
            training=TrainingConfig(
                batch_size=24,
                learning_rate=2e-4,
                warmup_steps=1000,
                max_steps=200000,
            ),
            model_name="aam-diffusion-small",
        ),
        "base": AamDiffusionConfig(
            model=ModelConfig(
                d_model=768,
                n_layers=12,
                n_heads=12,
                d_ff=3072,
                vocab_size=32000,
                max_seq_len=512,
            ),
            graph_encoder=GraphEncoderConfig(
                d_graph=512,
                n_graph_layers=4,
                n_graph_heads=8,
            ),
            diffusion=DiffusionConfig(
                n_timesteps=1000,
                n_inference_steps=50,
            ),
            training=TrainingConfig(
                batch_size=32,
                learning_rate=1e-4,
                warmup_steps=2000,
                max_steps=500000,
            ),
            model_name="aam-diffusion-base",
        ),
        "medium": AamDiffusionConfig(
            model=ModelConfig(
                d_model=1024,
                n_layers=12,
                n_heads=16,
                d_ff=4096,
                vocab_size=32000,
                max_seq_len=768,
            ),
            graph_encoder=GraphEncoderConfig(
                d_graph=768,
                n_graph_layers=6,
                n_graph_heads=12,
            ),
            diffusion=DiffusionConfig(
                n_timesteps=1000,
                n_inference_steps=50,
            ),
            training=TrainingConfig(
                batch_size=16,
                learning_rate=5e-5,
                warmup_steps=5000,
                max_steps=1000000,
            ),
            model_name="aam-diffusion-medium",
        ),
    }

    if model_size not in configs:
        raise ValueError(
            f"Unknown model_size '{model_size}'. "
            f"Choose from: {list(configs.keys())}"
        )

    return configs[model_size]
