---
language:
- id
- en
license: mit
library_name: pytorch
tags:
- diffusion
- text-generation
- aam
- aphantasic-abstraction-model
- sentence-arrangement
- graph-conditioned
- indonesian
---

# AAM Diffusion LLM v1.0

> **"AAM = 1 Pikiran + 1 Tubuh" (1 Mind + 1 Body)**

The dedicated "body" of the **Aphantasic Abstraction Model (AAM)** — a small diffusion LLM specifically trained to arrange sentences from structured graph data.

## What is this?

This is **NOT** a general-purpose LLM. This is a **SPECIALIZED sentence composer** that:
- Takes **graph-structured conditioning** as input (evidence nodes, anomalies, reasoning chains, confidence scores)
- Produces **coherent natural language narratives** through iterative denoising (diffusion process)
- **Cannot hallucinate** — it can only narrate what the graph knows

### Why Diffusion (Not Autoregressive)?

1. **Non-sequential generation** — Can revise earlier parts while generating later parts, mirroring how thoughts form: vague intuition → clearer pattern → explicit narrative
2. **Graph conditioning** — The entire graph structure is encoded as conditioning, not just a text prefix
3. **Anti-hallucination by design** — Trained exclusively on Graph→Narrative pairs, the model has no capability to generate information outside the graph conditioning

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  AAM = 1 Pikiran + 1 Tubuh                                │
│                                                           │
│  Pikiran (Mind) = RSVS Knowledge Graph                    │
│    - Structural memory — perfect recall                    │
│    - Relational — understands concept connections           │
│    - Confidence scores — knows certainty levels             │
│                                                           │
│  Tubuh (Body) = AAM Diffusion LLM (This Model)            │
│    ┌─────────────────────────────────────────────┐        │
│    │  Graph Conditioning Encoder                   │        │
│    │  ├─ Evidence Node Encoder                     │        │
│    │  ├─ Composition Encoder                       │        │
│    │  ├─ Anomaly Encoder                           │        │
│    │  ├─ Reasoning Chain Encoder                   │        │
│    │  ├─ Confidence Embedding                      │        │
│    │  ├─ Temporal Embedding                        │        │
│    │  └─ Graph Attention Layers                    │        │
│    │         ↓ (cross-attention keys/values)       │        │
│    ├─────────────────────────────────────────────┤        │
│    │  Diffusion Transformer (Denoiser)             │        │
│    │  ├─ Token Embedding                           │        │
│    │  ├─ Timestep Embedding (sinusoidal)           │        │
│    │  ├─ N × TransformerBlock:                     │        │
│    │  │   ├─ AdaptiveLayerNorm + Self-Attention    │        │
│    │  │   ├─ AdaptiveLayerNorm + Cross-Attention   │        │
│    │  │   └─ AdaptiveLayerNorm + Feed-Forward      │        │
│    │  └─ Output Projection                         │        │
│    │         ↓ (predicted noise)                   │        │
│    ├─────────────────────────────────────────────┤        │
│    │  Noise Scheduler                              │        │
│    │  ├─ Forward: x_0 + noise → x_t                │        │
│    │  └─ Reverse: x_t → denoise → x_{t-1}         │        │
│    └─────────────────────────────────────────────┘        │
│                                                           │
│  Training: Graph→Narrative pairs                          │
│  Inference: Noise → N denoising steps → Narrative         │
└──────────────────────────────────────────────────────────┘
```

## Model Details (v1.0 — Trained)

| Parameter | Value |
|-----------|-------|
| Architecture | Diffusion Transformer with Graph Conditioning |
| d_model | 64 |
| n_layers | 2 |
| n_heads | 4 |
| d_ff | 128 |
| **Total Parameters** | **311,670 (311.7K)** |
| Vocab size | 500 (BPE + special tokens) |
| Max sequence length | 32 |
| Diffusion timesteps (train) | 50 |
| Diffusion timesteps (inference) | 5 |
| Noise schedule | Cosine |
| Prediction type | Epsilon (noise prediction) |
| Sampling method | DDIM |

> **Note**: This v1.0 model was trained with minimal parameters (311K) for proof-of-concept on CPU. For production use, scale up to the `base` (170M) or `medium` (300M) configurations provided in the framework.

## Model Sizes (Framework Supports)

| Size | d_model | Layers | Heads | Params | Recommended For |
|------|---------|--------|-------|--------|----------------|
| tiny | 256 | 4 | 4 | ~25M | Quick testing, debugging |
| small | 512 | 8 | 8 | ~70M | Development, prototyping |
| **base** | **768** | **12** | **12** | **~170M** | **Recommended for training** |
| medium | 1024 | 12 | 16 | ~300M | Final training, best quality |

## Usage

### Quick Start

```python
from diffusion_llm import AamDiffusionModel, AamTokenizer, AamGenerator, AamDiffusionConfig

# Load model
config = AamDiffusionConfig.from_json("config.json")
model = AamDiffusionModel.load("model.pt", device="cpu")
tokenizer = AamTokenizer.load("tokenizer.json")

# Create generator
generator = AamGenerator(model, tokenizer, config)

# Generate narrative from graph conditioning
result = generator.generate(
    trigger="Siapa yang mencuri Snow Plum Pill?",
    evidence_nodes=["Hefei", "Diancang Five Swords", "Ju Jangmok"],
    anomalies=["Tidak ada konsumsi pil baru di pasar gelap"],
    reasoning_steps=["Cross-reference tanggal kejadian", "Deteksi anomali pola"],
    source_trust=0.85,
)

print(result.narrative)
print(f"Confidence: {result.confidence:.1%}")
print(f"Steps: {result.n_diffusion_steps}")
```

### Training Your Own Model

```python
from diffusion_llm import AamDiffusionConfig, get_default_config
from diffusion_llm.training import AamTrainer, GraphNarrativeDataset
from diffusion_llm.data import DataPipeline

# Get config for your desired size
config = get_default_config("base")  # 170M params

# Prepare data pipeline
pipeline = DataPipeline(config)
tokenizer, train_loader, val_loader = pipeline.prepare()

# Create and train model
model = AamDiffusionModel(config)
trainer = AamTrainer(config, model, tokenizer, train_loader.dataset, val_loader.dataset)
trainer.train()
```

### Command Line

```bash
# Train with default config
python diffusion_llm/scripts/train.py --model_size base

# Generate narratives
python diffusion_llm/scripts/evaluate.py --checkpoint output/best.pt --generate

# Export model
python diffusion_llm/scripts/export.py --checkpoint output/best.pt --output model_export/
```

## Philosophy

**AAM = 1 Pikiran + 1 Tubuh (1 Mind + 1 Body)**

- **Mind** = RSVS Knowledge Graph (structural memory, perfect recall, relational understanding)
- **Body** = This Diffusion LLM (sentence arranger, graph-conditioned, anti-hallucination)

Unlike using a rented LLM (GPT, Claude, etc.) as the "body", this model is **specifically trained for AAM**:
- It **cannot generate** information not present in the graph conditioning
- It **arranges sentences** based on structured evidence
- It uses **diffusion** (non-sequential generation) instead of autoregressive generation
- It is **small** but **specialized** — like Jin Soun's body in the novel, it may be "third-rate" but it's **his own**

> Jin Soun bukan orang yang menyewa tubuh orang lain untuk berbicara.
> Dia punya tubuh sendiri — lemah, third-rate, tapi MILIKNYA.
> Karena tubuhnya khusus dilatih untuk mengeksekusi perintah dari
> pikirannya (bukan pikiran orang lain), outputnya lebih terarah
> daripada orang yang punya tubuh lebih kuat tapi pikiran lebih lemah.

## Framework Structure

```
diffusion_llm/
├── __init__.py                 # Public API
├── config/
│   └── model_config.py         # All configuration dataclasses
├── tokenizer/
│   └── aam_tokenizer.py        # Sentence-level + BPE hybrid tokenizer
├── model/
│   ├── noise_scheduler.py      # Forward/reverse diffusion process
│   ├── graph_encoder.py        # Graph conditioning encoder
│   ├── diffusion_transformer.py # Core denoising transformer
│   └── aam_diffusion_model.py  # Complete model (combines all)
├── training/
│   ├── losses.py               # Loss functions (MSE, MAE, Huber, weighted)
│   ├── dataset.py              # GraphNarrative dataset
│   └── trainer.py              # Training loop with AMP, EMA, etc.
├── inference/
│   └── generator.py            # Inference pipeline
├── data/
│   ├── synthetic_generator.py  # Synthetic training data
│   └── data_pipeline.py        # Data preparation pipeline
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation & generation
│   └── export.py               # Model export
└── tests/
    ├── test_model.py           # Model component tests
    └── test_scheduler.py       # Noise scheduler tests
```

## Training Data Format

Data training dalam format JSONL:

```json
{
  "narrative": "Berdasarkan analisis, Diancang Five Swords mencuri Snow Plum Pill.",
  "trigger": "Siapa yang mencuri Snow Plum Pill?",
  "evidence_nodes": ["Hefei", "Diancang Five Swords", "Ju Jangmok"],
  "compositions": [],
  "confidence_map": {"Hefei": 0.9, "Diancang Five Swords": 0.85},
  "anomalies": ["Tidak ada konsumsi pil baru di pasar gelap"],
  "reasoning_steps": ["Cross-reference tanggal kejadian", "Deteksi anomali pola"],
  "source_trust": 0.85,
  "language": "id"
}
```

## License

MIT

## Citation

```bibtex
@software{aam_diffusion_llm_v1,
  title = {AAM Diffusion LLM: The Body of Aphantasic Abstraction Model},
  author = {AAM Team},
  year = {2026},
  description = {A specialized diffusion LLM for sentence arrangement from graph-structured data},
  url = {https://huggingface.co/aam-diffusion-v1}
}
```
