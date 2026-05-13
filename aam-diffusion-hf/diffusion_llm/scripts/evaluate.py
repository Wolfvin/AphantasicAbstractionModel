#!/usr/bin/env python3
"""
AAM Diffusion LLM — Evaluation Script

Evaluates a trained AAM Diffusion Model on test data or
generates sample narratives from graph conditioning.

Usage:
    # Evaluate on test data
    python scripts/evaluate.py --checkpoint output/best.pt

    # Generate sample narratives
    python scripts/evaluate.py --checkpoint output/best.pt --generate

    # Interactive mode
    python scripts/evaluate.py --checkpoint output/best.pt --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_llm.config.model_config import AamDiffusionConfig
from diffusion_llm.model.aam_diffusion_model import AamDiffusionModel
from diffusion_llm.tokenizer.aam_tokenizer import AamTokenizer
from diffusion_llm.inference.generator import AamGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AAM Diffusion LLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path")
    parser.add_argument("--generate", action="store_true", help="Generate sample narratives")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--test_data", type=str, default=None, help="Test data path (JSONL)")
    parser.add_argument("--n_steps", type=int, default=50, help="Inference denoising steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--language", type=str, default="id", help="Output language")
    return parser.parse_args()


def generate_samples(generator: AamGenerator, language: str) -> None:
    """Generate sample narratives from predefined graph conditioning."""
    samples = [
        {
            "trigger": "Siapa yang mencuri Snow Plum Pill?",
            "evidence_nodes": ["Hefei", "Diancang Five Swords", "Ju Jangmok", "Gyeryong Merchant Guild"],
            "anomalies": ["Tidak ada konsumsi pil baru di pasar gelap", "Pencuri menghilang tanpa jejak"],
            "reasoning_steps": ["Cross-reference tanggal kejadian", "Deteksi ketidaksesuaian pola"],
        },
        {
            "trigger": "Analisis pergerakan Diancang Five Swords",
            "evidence_nodes": ["Gu Ilmu", "Jang Hangi", "Diancang Five Swords", "Hefei"],
            "anomalies": ["Success rate pair lebih tinggi dari biasanya"],
            "reasoning_steps": ["Recall laporan terkait", "Pattern completion dari bukti"],
        },
        {
            "trigger": "Hubungan antara Ju Jangmok dan pencurian",
            "evidence_nodes": ["Ju Jangmok", "Snow Plum Pill", "dark_faction"],
            "anomalies": ["Ju Jangmok menghilang hari yang sama"],
            "reasoning_steps": ["Eliminasi tersangka obvious", "Verify konsistensi"],
        },
    ]

    print("\n" + "=" * 60)
    print("  AAM Diffusion LLM — Sample Generation")
    print("=" * 60)

    for i, sample in enumerate(samples, 1):
        result = generator.generate(
            trigger=sample["trigger"],
            evidence_nodes=sample["evidence_nodes"],
            anomalies=sample["anomalies"],
            reasoning_steps=sample["reasoning_steps"],
            language=language,
        )
        print(f"\n--- Sample {i} ---")
        print(f"Trigger: {sample['trigger']}")
        print(f"Evidence: {', '.join(sample['evidence_nodes'])}")
        print(f"Anomalies: {'; '.join(sample['anomalies'])}")
        print(f"\nGenerated Narrative:")
        print(result.narrative)
        print(f"\n[Steps: {result.n_diffusion_steps}, Time: {result.generation_time_s:.2f}s]")


def interactive_mode(generator: AamGenerator, language: str) -> None:
    """Interactive generation mode."""
    print("\n" + "=" * 60)
    print("  AAM Diffusion LLM — Interactive Mode")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        trigger = input("\nTrigger/Question: ").strip()
        if trigger.lower() in ("quit", "exit", "q"):
            break

        evidence = input("Evidence nodes (comma-separated): ").strip()
        evidence_nodes = [e.strip() for e in evidence.split(",") if e.strip()] if evidence else None

        anomalies_input = input("Anomalies (comma-separated): ").strip()
        anomalies = [a.strip() for a in anomalies_input.split(",") if a.strip()] if anomalies_input else None

        result = generator.generate(
            trigger=trigger,
            evidence_nodes=evidence_nodes,
            anomalies=anomalies,
            language=language,
        )
        print(f"\nGenerated Narrative:\n{result.narrative}")
        print(f"\n[Steps: {result.n_diffusion_steps}, Time: {result.generation_time_s:.2f}s, Confidence: {result.confidence:.1%}]")


def main() -> None:
    args = parse_args()

    # Load model
    logger.info("Loading model from %s", args.checkpoint)
    model = AamDiffusionModel.load(args.checkpoint)

    # Load or create tokenizer
    if args.tokenizer:
        tokenizer = AamTokenizer.load(args.tokenizer)
    else:
        # Try to find tokenizer in same directory as checkpoint
        tokenizer_path = Path(args.checkpoint).parent / "data" / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = AamTokenizer.load(tokenizer_path)
        else:
            logger.warning("No tokenizer found. Using untrained tokenizer.")
            tokenizer = AamTokenizer()

    # Create generator
    generator = AamGenerator(model, tokenizer, model.config)

    if args.interactive:
        interactive_mode(generator, args.language)
    elif args.generate:
        generate_samples(generator, args.language)
    else:
        logger.info("Use --generate or --interactive flag")


if __name__ == "__main__":
    main()
