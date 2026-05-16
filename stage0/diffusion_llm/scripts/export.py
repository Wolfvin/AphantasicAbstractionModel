#!/usr/bin/env python3
"""
AAM Diffusion LLM — Export Script

Export a trained model for deployment.

Usage:
    python scripts/export.py --checkpoint output/best.pt --output model_export/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_llm.config.model_config import AamDiffusionConfig
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AAM Diffusion Model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./model_export")
    parser.add_argument("--format", type=str, default="pt", choices=["pt", "onnx"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = AamDiffusionModel.load(args.checkpoint)
    model.eval()

    # Save model
    model_path = output_dir / "model.pt"
    model.save(str(model_path))
    logger.info("Model exported to %s", model_path)

    # Save config
    config_path = output_dir / "config.json"
    model.config.to_json(config_path)
    logger.info("Config saved to %s", config_path)

    # Try to copy tokenizer
    checkpoint_dir = Path(args.checkpoint).parent
    tokenizer_path = checkpoint_dir / "data" / "tokenizer.json"
    if tokenizer_path.exists():
        import shutil
        shutil.copy(tokenizer_path, output_dir / "tokenizer.json")
        logger.info("Tokenizer copied to %s", output_dir / "tokenizer.json")

    # Summary
    print(f"\nExport complete!")
    print(f"  Model: {model_path}")
    print(f"  Config: {config_path}")
    print(f"  Parameters: {model._format_params(model.get_num_params())}")
    print(f"\n  This is AAM's own body — 1 mind + 1 body.")
    print(f"  Mind = RSVS Knowledge Graph")
    print(f"  Body = This Diffusion Model ({model.config.model_name})")


if __name__ == "__main__":
    main()
