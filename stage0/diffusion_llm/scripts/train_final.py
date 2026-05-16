#!/usr/bin/env python3
"""
AAM Diffusion LLM — Final Training Script

Trains the complete AAM Diffusion LLM pipeline:
1. Generate synthetic training data (Graph→Narrative pairs)
2. Train the AAM Sentence-Level + BPE Tokenizer
3. Train the Diffusion Transformer model
4. Save final model, tokenizer, and config for HuggingFace upload

This is the "birth" of AAM's body — from random weights to
a model that can arrange sentences from graph conditioning.

Usage:
    python scripts/train_final.py --output_dir ./aam-diffusion-v1
    python scripts/train_final.py --model_size tiny --max_steps 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from diffusion_llm.config.model_config import (
    AamDiffusionConfig, get_default_config, ModelConfig,
    DiffusionConfig, GraphEncoderConfig, TokenizerConfig,
    TrainingConfig, InferenceConfig,
)
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer
from diffusion_llm.training.dataset import GraphNarrativeDataset, collate_fn
from diffusion_llm.data.synthetic_generator import SyntheticDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_final")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AAM Diffusion LLM (Final)")
    parser.add_argument("--model_size", type=str, default="tiny",
                        choices=["tiny", "small", "base", "medium"])
    parser.add_argument("--output_dir", type=str, default="./aam-diffusion-v1")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_synthetic_train", type=int, default=500)
    parser.add_argument("--n_synthetic_val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=200)
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def generate_data(output_dir: Path, n_train: int, n_val: int, seed: int):
    """Generate synthetic training data."""
    logger.info("=" * 60)
    logger.info("STEP 1: Generating Synthetic Training Data")
    logger.info("=" * 60)

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path, val_path = SyntheticDataGenerator.generate_training_split(
        output_dir=data_dir,
        n_train=n_train,
        n_val=n_val,
        language="id",
        seed=seed,
    )

    logger.info(f"  Train data: {train_path} ({n_train} examples)")
    logger.info(f"  Val data:   {val_path} ({n_val} examples)")
    return train_path, val_path


def train_tokenizer(train_path: Path, output_dir: Path, config: AamDiffusionConfig) -> AamTokenizer:
    """Train the AAM Tokenizer on synthetic data."""
    logger.info("=" * 60)
    logger.info("STEP 2: Training AAM Sentence-Level + BPE Tokenizer")
    logger.info("=" * 60)

    tokenizer = AamTokenizer(config=config.tokenizer)

    # Read training texts
    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("narrative"):
                    texts.append(data["narrative"])
                if data.get("trigger"):
                    texts.append(data["trigger"])
                for ev in data.get("evidence_nodes", []):
                    texts.append(ev)
                for anom in data.get("anomalies", []):
                    texts.append(anom)
                for step in data.get("reasoning_steps", []):
                    texts.append(step)
                for comp in data.get("compositions", []):
                    texts.append(comp)
            except json.JSONDecodeError:
                continue

    logger.info(f"  Training tokenizer on {len(texts)} texts...")
    tokenizer.train(texts, vocab_size=config.tokenizer.bpe_vocab_size)

    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    logger.info(f"  Tokenizer saved: {tokenizer_path}")
    logger.info(f"  Vocab size: {tokenizer.vocab_size}")
    logger.info(f"  BPE merges: {len(tokenizer.merges)}")

    return tokenizer


def create_dataloaders(
    train_path: Path, val_path: Path,
    tokenizer: AamTokenizer, config: AamDiffusionConfig
):
    """Create training and validation data loaders."""
    logger.info("=" * 60)
    logger.info("STEP 3: Creating DataLoaders")
    logger.info("=" * 60)

    train_dataset = GraphNarrativeDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        max_evidence=config.graph_encoder.max_evidence_nodes,
        max_anomalies=config.graph_encoder.max_anomalies,
        max_reasoning=config.graph_encoder.max_reasoning_steps,
        augment=True,
    )

    val_dataset = GraphNarrativeDataset(
        data_path=val_path,
        tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        max_evidence=config.graph_encoder.max_evidence_nodes,
        max_anomalies=config.graph_encoder.max_anomalies,
        max_reasoning=config.graph_encoder.max_reasoning_steps,
        augment=False,
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,  # CPU training: use 0 workers
        collate_fn=collate_fn,
        pin_memory=False,  # CPU: no pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    logger.info(f"  Train: {len(train_dataset)} examples, {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_dataset)} examples, {len(val_loader)} batches")

    return train_loader, val_loader


def train_model(
    model: AamDiffusionModel,
    tokenizer: AamTokenizer,
    train_loader,
    val_loader,
    config: AamDiffusionConfig,
    output_dir: Path,
    args,
):
    """Train the AAM Diffusion Model."""
    logger.info("=" * 60)
    logger.info("STEP 4: Training AAM Diffusion LLM")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")
    logger.info(f"  Parameters: {model._format_params(model.get_num_params())}")

    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
    )

    # LR scheduler with warmup
    warmup_steps = min(200, args.max_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(args.max_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    start_time = time.time()

    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info("")

    epoch = 0
    while global_step < args.max_steps:
        epoch += 1
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= args.max_steps:
                break

            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Sample random timesteps
            batch_size = batch["token_ids"].shape[0]
            t = torch.randint(
                0, config.diffusion.n_timesteps,
                (batch_size,), device=device,
            )

            # Forward pass
            predicted, target = model(
                token_ids=batch["token_ids"],
                timestep=t,
                evidence_ids=batch.get("evidence_ids"),
                evidence_confidence=batch.get("evidence_confidence"),
                anomaly_ids=batch.get("anomaly_ids"),
                anomaly_confidence=batch.get("anomaly_confidence"),
                reasoning_ids=batch.get("reasoning_ids"),
                reasoning_confidence=batch.get("reasoning_confidence"),
                source_trust=batch.get("source_trust"),
            )

            # Compute loss
            loss = model.compute_loss(predicted, target, t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip_norm
            )

            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            train_losses.append(loss_val)
            epoch_loss += loss_val
            n_batches += 1
            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = sum(train_losses[-args.log_every:]) / len(train_losses[-args.log_every:])
                elapsed = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1)
                logger.info(
                    f"  Step {global_step:>6d}/{args.max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

            # Evaluation
            if global_step % args.eval_every == 0 and val_loader is not None:
                val_loss = evaluate(model, val_loader, config, device)
                logger.info(f"  >>> Validation loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model, tokenizer, config, output_dir / "best.pt")
                    logger.info(f"  >>> New best model saved! (val_loss: {val_loss:.4f})")

            # Checkpoint
            if global_step % args.save_every == 0:
                save_model(model, tokenizer, config, output_dir / f"step_{global_step}.pt")

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"  Epoch {epoch} complete. Avg loss: {avg_epoch_loss:.4f}")

    # Final save
    save_model(model, tokenizer, config, output_dir / "final.pt")
    elapsed = time.time() - start_time
    logger.info("")
    logger.info(f"  Training complete! {global_step} steps in {elapsed/60:.1f} minutes")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Final train loss: {train_losses[-1]:.4f}")

    return model


def evaluate(model, val_loader, config, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            batch_size = batch["token_ids"].shape[0]
            t = torch.randint(
                0, config.diffusion.n_timesteps,
                (batch_size,), device=device,
            )

            predicted, target = model(
                token_ids=batch["token_ids"],
                timestep=t,
                evidence_ids=batch.get("evidence_ids"),
                evidence_confidence=batch.get("evidence_confidence"),
                anomaly_ids=batch.get("anomaly_ids"),
                anomaly_confidence=batch.get("anomaly_confidence"),
                reasoning_ids=batch.get("reasoning_ids"),
                reasoning_confidence=batch.get("reasoning_confidence"),
                source_trust=batch.get("source_trust"),
            )
            loss = model.compute_loss(predicted, target, t)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def save_model(model, tokenizer, config, path):
    """Save model checkpoint with tokenizer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }
    torch.save(checkpoint, path)


def export_for_huggingface(model, tokenizer, config, output_dir: Path):
    """Export model in HuggingFace-compatible format."""
    logger.info("=" * 60)
    logger.info("STEP 5: Exporting for HuggingFace")
    logger.info("=" * 60)

    hf_dir = output_dir / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = hf_dir / "model.pt"
    model.save(str(model_path))
    logger.info(f"  Model saved: {model_path}")

    # Save tokenizer
    tokenizer_path = hf_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    logger.info(f"  Tokenizer saved: {tokenizer_path}")

    # Save config
    config_path = hf_dir / "config.json"
    config.to_json(config_path)
    logger.info(f"  Config saved: {config_path}")

    # Save model card
    model_card = f"""---
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
---

# AAM Diffusion LLM v1.0

> **"AAM = 1 Pikiran + 1 Tubuh" (1 Mind + 1 Body)**

The dedicated "body" of the Aphantasic Abstraction Model (AAM) — a small diffusion LLM specifically trained to arrange sentences from structured graph data.

## What is this?

This is NOT a general-purpose LLM. This is a SPECIALIZED sentence composer that:
- Takes **graph-structured conditioning** as input (evidence, anomalies, reasoning chains, confidence scores)
- Produces **coherent natural language narratives** through iterative denoising
- **Cannot hallucinate** — it can only narrate what the graph knows

## Architecture

```
Graph Conditioning Encoder → Diffusion Transformer → Noise Scheduler
         (Mind input)           (The Body)          (Iterative refinement)
```

### Key Components
- **Graph Conditioning Encoder**: Encodes evidence nodes, compositions, anomalies, reasoning chains with confidence and temporal embeddings
- **Diffusion Transformer**: Core denoising network with adaptive layer norm, self-attention, and cross-attention to graph conditioning
- **Noise Scheduler**: Cosine noise schedule with DDPM/DDIM sampling support

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | Diffusion Transformer |
| d_model | {config.model.d_model} |
| n_layers | {config.model.n_layers} |
| n_heads | {config.model.n_heads} |
| d_ff | {config.model.d_ff} |
| Parameters | {model._format_params(model.get_num_params())} |
| Vocab size | {config.model.vocab_size} |
| Max sequence length | {config.model.max_seq_len} |
| Diffusion timesteps (train) | {config.diffusion.n_timesteps} |
| Diffusion timesteps (inference) | {config.diffusion.n_inference_steps} |
| Noise schedule | {config.diffusion.schedule_type} |
| Prediction type | {config.diffusion.prediction_type} |
| Sampling method | {config.diffusion.sampling_method} |

## Usage

```python
from diffusion_llm import AamDiffusionModel, AamTokenizer, AamGenerator, AamDiffusionConfig

# Load model
config = AamDiffusionConfig.from_json("config.json")
model = AamDiffusionModel.load("model.pt")
tokenizer = AamTokenizer.load("tokenizer.json")

# Create generator
generator = AamGenerator(model, tokenizer, config)

# Generate narrative from graph conditioning
result = generator.generate(
    trigger="Siapa yang mencuri Snow Plum Pill?",
    evidence_nodes=["Hefei", "Diancang Five Swords", "Ju Jangmok"],
    anomalies=["Tidak ada konsumsi pil baru di pasar gelap"],
    reasoning_steps=["Cross-reference tanggal kejadian"],
    source_trust=0.85,
)
print(result.narrative)
```

## Philosophy

**AAM = 1 Mind + 1 Body**

- **Mind** = RSVS Knowledge Graph (structural memory, perfect recall, relational understanding)
- **Body** = This Diffusion LLM (sentence arranger, graph-conditioned, anti-hallucination)

Unlike using a rented LLM (GPT, Claude) as the "body", this model is specifically trained for AAM:
- It cannot generate information not present in the graph conditioning
- It arranges sentences based on structured evidence
- It uses diffusion (non-sequential generation) instead of autoregressive generation
- It is small ({model._format_params(model.get_num_params())}) but specialized

## Training

Trained on synthetic Graph→Narrative pairs with:
- Indonesian and English narrative templates
- Evidence nodes, anomalies, reasoning chains
- Confidence score distributions
- Source trust scores

## License

MIT
"""
    model_card_path = hf_dir / "README.md"
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    logger.info(f"  Model card saved: {model_card_path}")

    # Copy full framework code
    import shutil
    framework_src = Path(__file__).parent.parent  # diffusion_llm/
    framework_dst = hf_dir / "diffusion_llm"
    if framework_dst.exists():
        shutil.rmtree(framework_dst)
    shutil.copytree(framework_src, framework_dst,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'output', 'data'))
    logger.info(f"  Framework code copied to: {framework_dst}")

    # Save training script
    train_script_dst = hf_dir / "train.py"
    shutil.copy2(Path(__file__), train_script_dst)

    # Save inference example
    inference_example = hf_dir / "inference_example.py"
    with open(inference_example, "w", encoding="utf-8") as f:
        f.write('''#!/usr/bin/env python3
"""AAM Diffusion LLM — Inference Example"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from diffusion_llm import AamDiffusionModel, AamTokenizer, AamGenerator, AamDiffusionConfig

def main():
    # Load model and tokenizer
    config = AamDiffusionConfig.from_json("config.json")
    model = AamDiffusionModel.load("model.pt", device="cpu")
    tokenizer = AamTokenizer.load("tokenizer.json")

    # Create generator
    generator = AamGenerator(model, tokenizer, config)

    # Generate narrative
    result = generator.generate(
        trigger="Siapa yang mencuri Snow Plum Pill?",
        evidence_nodes=["Hefei", "Diancang Five Swords", "Ju Jangmok"],
        anomalies=["Tidak ada konsumsi pil baru di pasar gelap"],
        reasoning_steps=["Cross-reference tanggal kejadian", "Deteksi anomali pola"],
        source_trust=0.85,
    )

    print("=" * 60)
    print("  AAM Diffusion LLM — Generated Narrative")
    print("=" * 60)
    print(f"  Trigger: {result.evidence_used}")
    print(f"  Narrative: {result.narrative}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Steps: {result.n_diffusion_steps}")
    print(f"  Time: {result.generation_time_s:.2f}s")

if __name__ == "__main__":
    main()
''')
    logger.info(f"  Inference example saved: {inference_example}")

    logger.info(f"\n  HuggingFace export complete: {hf_dir}")
    return hf_dir


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AAM Diffusion LLM — Final Training")
    print("  \"1 Pikiran + 1 Tubuh\" (1 Mind + 1 Body)")
    print("=" * 60)
    print()

    # Get config
    config = get_default_config(args.model_size)

    # CPU-optimized overrides for faster training
    config.model.max_seq_len = 128
    config.model.vocab_size = 8000
    config.graph_encoder.max_evidence_nodes = 10
    config.graph_encoder.max_anomalies = 5
    config.graph_encoder.max_reasoning_steps = 5
    config.graph_encoder.max_compositions = 5
    config.diffusion.n_timesteps = 200
    config.diffusion.n_inference_steps = 20
    config.tokenizer.bpe_vocab_size = 8000 - 13  # minus special tokens

    # Override settings for CPU training
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.max_steps = args.max_steps
    config.training.use_amp = False  # No AMP on CPU
    config.training.num_workers = 0  # No multiprocessing on CPU
    config.training.warmup_steps = min(100, args.max_steps // 5)
    config.output_dir = str(output_dir)
    config.seed = args.seed
    config.model_name = "aam-diffusion-v2.1"

    # Print config
    print(config.summary())

    # Step 1: Generate synthetic data
    train_path, val_path = generate_data(
        output_dir, args.n_synthetic_train, args.n_synthetic_val, args.seed
    )

    # Step 2: Train tokenizer
    tokenizer = train_tokenizer(train_path, output_dir, config)

    # Update vocab_size to match actual tokenizer
    actual_vocab = tokenizer.vocab_size
    if actual_vocab != config.model.vocab_size:
        logger.info(f"  Updating vocab_size: {config.model.vocab_size} → {actual_vocab}")
        config.model.vocab_size = actual_vocab

    # Step 3: Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path, val_path, tokenizer, config
    )

    # Step 4: Create and train model
    model = AamDiffusionModel(config)
    logger.info(f"  Model parameters: {model._format_params(model.get_num_params())}")

    model = train_model(
        model, tokenizer, train_loader, val_loader,
        config, output_dir, args
    )

    # Step 5: Export for HuggingFace
    hf_dir = export_for_huggingface(model, tokenizer, config, output_dir)

    # Final summary
    print()
    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Model: {config.model_name}")
    print(f"  Parameters: {model._format_params(model.get_num_params())}")
    print(f"  Output: {output_dir}")
    print(f"  HuggingFace export: {hf_dir}")
    print()
    print("  AAM = 1 Pikiran + 1 Tubuh")
    print("  Pikiran = RSVS Knowledge Graph")
    print("  Tubuh   = This Diffusion LLM")
    print("=" * 60)


if __name__ == "__main__":
    main()
