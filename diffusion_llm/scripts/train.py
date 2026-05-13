#!/usr/bin/env python3
"""
AAM Diffusion LLM — Training Script

Main entry point for training the AAM Diffusion Model.

Usage:
    # Train with default config (base model)
    python scripts/train.py

    # Train with specific model size
    python scripts/train.py --model_size small

    # Train with custom config
    python scripts/train.py --config path/to/config.json

    # Train with specific data
    python scripts/train.py --train_data path/to/train.jsonl --val_data path/to/val.jsonl

Analogi: Seperti Jin Soun memulai latihan fisiknya —
ini adalah titik awal di mana "tubuh" AAM mulai dilatih.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_llm.config.model_config import AamDiffusionConfig, get_default_config
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer
from diffusion_llm.training.trainer import AamTrainer
from diffusion_llm.training.dataset import GraphNarrativeDataset
from diffusion_llm.data.data_pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AAM Diffusion LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_size", type=str, default="base",
        choices=["tiny", "small", "base", "medium"],
        help="Model size preset",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom config JSON (overrides --model_size)",
    )

    # Data
    parser.add_argument(
        "--train_data", type=str, default=None,
        help="Path to training data (JSONL)",
    )
    parser.add_argument(
        "--val_data", type=str, default=None,
        help="Path to validation data (JSONL)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--force_regenerate", action="store_true",
        help="Force regenerate synthetic data",
    )

    # Training overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--n_timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Load or create config
    if args.config:
        config = AamDiffusionConfig.from_json(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        config = get_default_config(args.model_size)
        logger.info("Using %s model config", args.model_size)

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.train_data:
        config.training.train_data_path = args.train_data
    if args.val_data:
        config.training.val_data_path = args.val_data
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.n_timesteps:
        config.diffusion.n_timesteps = args.n_timesteps
    config.seed = args.seed

    # Print config summary
    print(config.summary())

    # Save config
    config_path = Path(config.output_dir) / "config.json"
    config.to_json(config_path)
    logger.info("Config saved to %s", config_path)

    # Step 1: Prepare data
    pipeline = DataPipeline(config)
    tokenizer, train_loader, val_loader = pipeline.prepare(
        force_regenerate=args.force_regenerate,
    )

    # Step 2: Create model
    model = AamDiffusionModel(config)
    logger.info(
        "Model created: %s parameters",
        model._format_params(model.get_num_params()),
    )

    # Step 3: Create datasets (using pre-created loaders)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset if val_loader else None

    # Step 4: Create trainer and train
    trainer = AamTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Override data loaders (already created by pipeline)
    trainer.train_loader = train_loader
    trainer.val_loader = val_loader

    # Start training
    trainer.train()

    logger.info("Training complete! Output saved to %s", config.output_dir)


if __name__ == "__main__":
    main()
