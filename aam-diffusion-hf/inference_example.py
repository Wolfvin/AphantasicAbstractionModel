#!/usr/bin/env python3
"""AAM Diffusion LLM v1.0 — Inference Example"""

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

    # Generate narrative from graph conditioning
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
    print(f"  Narrative: {result.narrative}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Steps: {result.n_diffusion_steps}")
    print(f"  Time: {result.generation_time_s:.2f}s")

if __name__ == "__main__":
    main()
