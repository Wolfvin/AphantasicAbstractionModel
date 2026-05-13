#!/usr/bin/env python3
"""
AAM Diffusion LLM — Minimal Training Script for CPU

Trains a very small AAM Diffusion LLM model on CPU.
"""

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

def main():
    from diffusion_llm.config.model_config import (
        AamDiffusionConfig, ModelConfig, DiffusionConfig,
        GraphEncoderConfig, TokenizerConfig, TrainingConfig, InferenceConfig,
    )
    from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
    from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer
    from diffusion_llm.training.dataset import GraphNarrativeDataset, collate_fn
    from diffusion_llm.data.synthetic_generator import SyntheticDataGenerator
    from torch.utils.data import DataLoader

    output_dir = Path("./aam-diffusion-v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ===== STEP 1: Generate Data =====
    logger.info("STEP 1: Generating synthetic data...")
    train_path, val_path = SyntheticDataGenerator.generate_training_split(
        output_dir=data_dir, n_train=200, n_val=20, language="id", seed=42,
    )

    # ===== STEP 2: Train Tokenizer =====
    logger.info("STEP 2: Training tokenizer...")
    tokenizer = AamTokenizer()

    texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                for key in ["narrative", "trigger"]:
                    if data.get(key):
                        texts.append(data[key])
                for key in ["evidence_nodes", "anomalies", "reasoning_steps"]:
                    for item in data.get(key, []):
                        texts.append(item)
            except json.JSONDecodeError:
                continue

    tokenizer.train(texts, vocab_size=2000)
    tokenizer.save(data_dir / "tokenizer.json")
    actual_vocab = tokenizer.vocab_size
    logger.info(f"  Tokenizer: vocab_size={actual_vocab}, merges={len(tokenizer.merges)}")

    # ===== STEP 3: Config =====
    config = AamDiffusionConfig(
        model=ModelConfig(
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            vocab_size=actual_vocab,
            max_seq_len=64,
            pos_encoding_type="learned",
            use_flash_attention=False,
            norm_type="layernorm",
            init_std=0.02,
        ),
        diffusion=DiffusionConfig(
            n_timesteps=100,
            n_inference_steps=10,
            schedule_type="cosine",
            prediction_type="epsilon",
            loss_type="mse",
            loss_weighting="none",
        ),
        graph_encoder=GraphEncoderConfig(
            d_graph=64,
            n_graph_layers=1,
            n_graph_heads=2,
            max_evidence_nodes=5,
            max_compositions=3,
            max_anomalies=3,
            max_reasoning_steps=3,
            conditioning_method="cross_attention",
            embed_confidence=False,
            embed_temporal=False,
        ),
        tokenizer=TokenizerConfig(bpe_vocab_size=2000),
        training=TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            max_steps=100,
            warmup_steps=10,
            use_amp=False,
            num_workers=0,
            grad_clip_norm=1.0,
        ),
        inference=InferenceConfig(n_steps=10),
        model_name="aam-diffusion-v1.0",
        output_dir=str(output_dir),
        seed=42,
    )

    # ===== STEP 4: Create Model =====
    logger.info("STEP 3: Creating model...")
    model = AamDiffusionModel(config)
    n_params = model.get_num_params()
    logger.info(f"  Parameters: {model._format_params(n_params)} ({n_params:,})")

    # ===== STEP 5: Create DataLoaders =====
    logger.info("STEP 4: Creating dataloaders...")
    train_dataset = GraphNarrativeDataset(
        data_path=train_path, tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        max_evidence=config.graph_encoder.max_evidence_nodes,
        max_anomalies=config.graph_encoder.max_anomalies,
        max_reasoning=config.graph_encoder.max_reasoning_steps,
        augment=True,
    )
    val_dataset = GraphNarrativeDataset(
        data_path=val_path, tokenizer=tokenizer,
        max_seq_len=config.model.max_seq_len,
        max_evidence=config.graph_encoder.max_evidence_nodes,
        max_anomalies=config.graph_encoder.max_anomalies,
        max_reasoning=config.graph_encoder.max_reasoning_steps,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # ===== STEP 6: Train =====
    logger.info("STEP 5: Training...")
    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    max_steps = 100

    start_time = time.time()
    global_step = 0
    train_losses = []

    for epoch in range(50):  # Max epochs
        model.train()
        for batch in train_loader:
            if global_step >= max_steps:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            batch_size = batch["token_ids"].shape[0]
            t = torch.randint(0, config.diffusion.n_timesteps, (batch_size,), device=device)

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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

            if global_step % 10 == 0:
                avg = sum(train_losses[-10:]) / len(train_losses[-10:])
                elapsed = time.time() - start_time
                logger.info(f"  Step {global_step}/{max_steps} | Loss: {avg:.4f} | Time: {elapsed:.1f}s")

        if global_step >= max_steps:
            break

    # ===== STEP 7: Evaluate =====
    logger.info("STEP 6: Evaluating...")
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            batch_size = batch["token_ids"].shape[0]
            t = torch.randint(0, config.diffusion.n_timesteps, (batch_size,), device=device)

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
            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
    logger.info(f"  Val loss: {avg_val_loss:.4f}")

    # ===== STEP 8: Save =====
    logger.info("STEP 7: Saving model...")

    # Save model
    model_path = output_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, model_path)

    # Save tokenizer (already saved)
    # Save config
    config.to_json(output_dir / "config.json")

    elapsed = time.time() - start_time
    logger.info(f"\n  DONE! {global_step} steps in {elapsed:.1f}s")
    logger.info(f"  Final train loss: {train_losses[-1]:.4f}")
    logger.info(f"  Val loss: {avg_val_loss:.4f}")
    logger.info(f"  Parameters: {model._format_params(n_params)}")
    logger.info(f"  Output: {output_dir}")

    return model, tokenizer, config, output_dir


if __name__ == "__main__":
    main()
