# AAM Diffusion LLM Framework

> **"AAM = 1 Pikiran + 1 Tubuh" (1 Mind + 1 Body)**

Framework khusus untuk melatih Diffusion LLM yang menjadi "tubuh" (body) dari Aphantasic Abstraction Model (AAM). Ini BUKAN LLM umum — ini model yang KHUSUS dilatih untuk menyusun kalimat dari data graph yang terstruktur.

---

## Filosofi

### Kenapa Bukan LLM Umum?

Konsep sebelumnya: "tubuh Jin Soun = LLM umum (GPT, Claude, dll.)" — ini **salah besar**.

| Aspek | LLM Umum (Sewaan) | AAM Diffusion LLM (Milik Sendiri) |
|-------|-------------------|-----------------------------------|
| Input | Prompt teks | Graph conditioning (evidence, anomaly, dll.) |
| Output | Teks probabilistik | Narrative yang grounded di graph |
| Hallucination | BISA mengarang | TIDAK BISA — hanya menarasikan apa yang graph ketahui |
| Tujuan | General purpose | Khusus menyusun kalimat dari graph |
| Ukuran | 7B-175B params | 100M-500M params |
| Metode | Autoregressive | Diffusion (non-sequential) |
| Identitas | Sewaan | MILIK AAM sendiri |

### Kenapa Diffusion (Bukan Autoregressive)?

1. **Non-sequential** — Bisa merevisi bagian awal saat generating bagian akhir. Mirip cara Jin Soun membentuk pikiran: vague → clearer → explicit.

2. **Graph conditioning** — Seluruh graph bisa di-encode sebagai conditioning, bukan hanya prefix. Autoregressive hanya bisa melihat "apa yang sudah di-generate sebelumnya."

3. **Coherent long-form** — Diffusion menghasilkan teks yang lebih koheren untuk narasi panjang karena setiap token "mengetahui" tentang token lain.

4. **Anti-hallucination** — Model dilatih KHUSUS untuk Graph→Narrative, tidak punya kapabilitas mengarang informasi di luar graph.

---

## Arsitektur

```
┌──────────────────────────────────────────────────────────┐
│  AAM = 1 Pikiran + 1 Tubuh                                │
│                                                           │
│  Pikiran (Mind) = RSVS Knowledge Graph                    │
│    - Structural memory — mengingat SEMUA                  │
│    - Relational — memahami koneksi antar konsep           │
│    - Perfect recall — tidak pernah lupa                   │
│    - Confidence scores — tahu apa yang pasti vs ragu      │
│                                                           │
│  Tubuh (Body) = AAM Diffusion LLM                         │
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

---

## Struktur Folder

```
diffusion_llm/
├── __init__.py                 # Package init with public API
├── config/
│   ├── __init__.py
│   └── model_config.py         # All configuration dataclasses
├── tokenizer/
│   ├── __init__.py
│   └── aam_tokenizer.py        # Sentence-level + BPE hybrid tokenizer
├── model/
│   ├── __init__.py
│   ├── noise_scheduler.py      # Forward/reverse diffusion process
│   ├── graph_encoder.py        # Graph conditioning encoder
│   ├── diffusion_transformer.py # Core denoising transformer
│   └── aam_diffusion_model.py  # Complete model (combines all)
├── training/
│   ├── __init__.py
│   ├── losses.py               # Loss functions (MSE, MAE, Huber, weighted)
│   ├── dataset.py              # GraphNarrative dataset
│   └── trainer.py              # Training loop with AMP, EMA, etc.
├── inference/
│   ├── __init__.py
│   └── generator.py            # Inference pipeline
├── data/
│   ├── __init__.py
│   ├── synthetic_generator.py  # Synthetic training data
│   └── data_pipeline.py        # Data preparation pipeline
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation & generation
│   └── export.py               # Model export
├── tests/
│   ├── __init__.py
│   ├── test_scheduler.py       # Noise scheduler tests
│   └── test_model.py           # Model component tests
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pytest
```

### 2. Generate Synthetic Data

```python
from diffusion_llm.data.synthetic_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42, language="id")
train_path, val_path = generator.generate_training_split(
    output_dir="./data",
    n_train=10000,
    n_val=500,
)
```

### 3. Train the Model

```bash
# Quick test with tiny model
python diffusion_llm/scripts/train.py --model_size tiny --max_steps 100

# Full training with base model
python diffusion_llm/scripts/train.py --model_size base --max_steps 500000
```

### 4. Generate Narratives

```bash
# Generate samples
python diffusion_llm/scripts/evaluate.py --checkpoint output/best.pt --generate

# Interactive mode
python diffusion_llm/scripts/evaluate.py --checkpoint output/best.pt --interactive
```

### 5. Programmatic Usage

```python
from diffusion_llm import (
    AamDiffusionConfig, get_default_config,
    AamDiffusionModel, AamTokenizer, AamGenerator,
)

# Load model and tokenizer
config = AamDiffusionConfig.from_json("output/config.json")
model = AamDiffusionModel.load("output/best.pt")
tokenizer = AamTokenizer.load("output/data/tokenizer.json")

# Create generator
generator = AamGenerator(model, tokenizer, config)

# Generate narrative from graph conditioning
result = generator.generate(
    trigger="Siapa yang mencuri Snow Plum Pill?",
    evidence_nodes=["Hefei", "Diancang Five Swords", "Ju Jangmok"],
    anomalies=["Tidak ada konsumsi pil baru di pasar gelap"],
    reasoning_steps=["Cross-reference tanggal kejadian", "Deteksi anomali"],
    source_trust=0.85,
)

print(result.narrative)
print(f"Confidence: {result.confidence:.1%}")
print(f"Steps: {result.n_diffusion_steps}")
```

---

## Model Sizes

| Size | d_model | Layers | Heads | Params | Recommended For |
|------|---------|--------|-------|--------|----------------|
| tiny | 256 | 4 | 4 | ~25M | Quick testing, debugging |
| small | 512 | 8 | 8 | ~70M | Development, prototyping |
| **base** | **768** | **12** | **12** | **~170M** | **Recommended for training** |
| medium | 1024 | 12 | 16 | ~300M | Final training, best quality |

---

## Konfigurasi

### Model Config

```python
from diffusion_llm.config.model_config import AamDiffusionConfig, ModelConfig, DiffusionConfig

config = AamDiffusionConfig(
    model=ModelConfig(
        d_model=768,        # Hidden dimension
        n_layers=12,        # Transformer blocks
        n_heads=12,         # Attention heads
        d_ff=3072,          # Feed-forward dimension
        vocab_size=32000,   # Vocabulary size
        max_seq_len=512,    # Maximum sequence length
    ),
    diffusion=DiffusionConfig(
        n_timesteps=1000,   # Training timesteps
        n_inference_steps=50,  # Inference steps (fewer = faster)
        schedule_type="cosine",  # Noise schedule
        prediction_type="epsilon",  # Predict noise
        sampling_method="ddim",  # Fast deterministic sampling
    ),
)
```

### Inference Config

```python
from diffusion_llm.config.model_config import InferenceConfig

inference = InferenceConfig(
    n_steps=50,           # Denoising steps
    temperature=1.0,      # Sampling temperature
    top_k=50,             # Top-k sampling
    max_output_sentences=16,  # Max sentences
    language="id",        # Output language
)
```

---

## Integrasi dengan AAM Pipeline

Framework ini dirancang untuk menjadi "tubuh" dari AAM. Setelah model dilatih,
integrasi dengan `pipeline.py` sangat mudah:

```python
# Dalam pipeline.py, ganti fallback:
from diffusion_llm import AamDiffusionModel, AamTokenizer, AamGenerator

class AamPipeline:
    def __init__(self, ...):
        # Load trained diffusion model
        diffusion_config = AamDiffusionConfig.from_json("path/to/config.json")
        diffusion_model = AamDiffusionModel.load("path/to/best.pt")
        diffusion_tokenizer = AamTokenizer.load("path/to/tokenizer.json")
        self.diffusion_llm = AamGenerator(diffusion_model, diffusion_tokenizer, diffusion_config)
```

---

## Training Data Format

Data training dalam format JSONL, satu contoh per baris:

```json
{
  "narrative": "Berdasarkan analisis, Diancang Five Swords mencuri Snow Plum Pill menggunakan Ju Jangmok sebagai kambing hitam.",
  "trigger": "Siapa yang mencuri Snow Plum Pill?",
  "evidence_nodes": ["Hefei", "Diancang Five Swords", "Ju Jangmok", "Gyeryong Merchant Guild"],
  "compositions": [],
  "confidence_map": {"Hefei": 0.9, "Diancang Five Swords": 0.85, "Ju Jangmok": 0.7},
  "anomalies": ["Tidak ada konsumsi pil baru di pasar gelap", "Pencuri menghilang tanpa jejak"],
  "reasoning_steps": ["Cross-reference tanggal kejadian", "Deteksi ketidaksesuaian pola", "Pattern completion dari bukti terpisah"],
  "source_trust": 0.85,
  "temporal_context": [],
  "language": "id",
  "source": "synthetic"
}
```

---

## Running Tests

```bash
# Run all tests
cd diffusion_llm
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_model.py -v

# Run with coverage
python -m pytest tests/ --cov=diffusion_llm
```

---

## Roadmap

- [x] **Phase 1: Framework Design** — Arsitektur, config, interface
- [x] **Phase 2: Core Components** — Noise scheduler, transformer, graph encoder, tokenizer
- [x] **Phase 3: Training Infrastructure** — Trainer, dataset, loss functions, synthetic data
- [x] **Phase 4: Inference Pipeline** — Generator, batch generation, interactive mode
- [ ] **Phase 5: Training Execution** — Train on synthetic data, iterate
- [ ] **Phase 6: Real Data** — Collect real Graph→Narrative pairs from AAM usage
- [ ] **Phase 7: Optimization** — Quantization, distillation, flash attention
- [ ] **Phase 8: Integration** — Plug trained model into AAM pipeline

---

## Analogi Novel

> Jin Soun bukan orang yang menyewa tubuh orang lain untuk berbicara.
> Dia punya tubuh sendiri — lemah, third-rate, tapi MILIKNYA.
> Karena tubuhnya khusus dilatih untuk mengeksekusi perintah dari
> pikirannya (bukan pikiran orang lain), outputnya lebih terarah
> daripada orang yang punya tubuh lebih kuat tapi pikiran lebih lemah.
>
> **AAM = 1 pikiran + 1 tubuh. Bukan 1 pikiran + tubuh sewaan.**
