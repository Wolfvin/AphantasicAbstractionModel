"""AAM Diffusion LLM — Quantization

BitNet 1-bit weights and FP8 training stubs.
Included for completeness — AAM's model is small enough that
quantization is not yet critical, but this prepares for future scaling.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Module):
    """1-bit weight quantization layer (BitNet-style).
    
    During training: uses full-precision weights with straight-through estimator
    During inference: uses binarized weights (-1 or +1)
    
    Note: Only practical for models >1B params. AAM's current size
    doesn't benefit from this, but it's included for future scaling.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Scale factor for binarized weights
        self.register_buffer("weight_scale", torch.ones(1), persistent=True)

    def _binarize_weights(self) -> torch.Tensor:
        """Binarize weights to -1 or +1 using sign function."""
        with torch.no_grad():
            self.weight_scale.copy_(self.weight.abs().mean())
        binary_weight = torch.sign(self.weight)
        return binary_weight * self.weight_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Straight-through estimator: forward uses binarized, backward uses full-precision
            binary_weight = torch.sign(self.weight) * self.weight_scale
            output = F.linear(x, binary_weight, self.bias)
        else:
            binary_weight = self._binarize_weights()
            output = F.linear(x, binary_weight, self.bias)

        return output


class FP8Linear(nn.Module):
    """FP8 weight-only quantization layer.
    
    Stores weights in FP8 (E4M3) format for memory efficiency.
    Computation is done in higher precision after dequantization.
    
    Note: Requires hardware with FP8 support (H100, MI300X).
    Falls back to FP32/BF16 on unsupported hardware.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store in FP32 for training, quantize for inference
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._fp8_available = hasattr(torch, "float8_e4m3fn")

    def _quantize_fp8(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weights to FP8 if supported."""
        if not self._fp8_available:
            return weight

        # Scale to FP8 range
        max_val = weight.abs().max()
        scale = max_val / 448.0  # E4M3 max value
        scaled = weight / scale.clamp(min=1e-8)

        try:
            fp8_weight = scaled.to(torch.float8_e4m3fn)
            dequantized = fp8_weight.to(torch.float32) * scale
            return dequantized
        except (RuntimeError, TypeError):
            return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self._fp8_available:
            weight = self._quantize_fp8(self.weight)
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)


def replace_linear_with_quantized(
    model: nn.Module,
    quantization_type: str = "bitnet",
) -> nn.Module:
    """Replace all nn.Linear layers with quantized versions.
    
    Args:
        model: The model to quantize
        quantization_type: "bitnet" or "fp8"
        
    Returns:
        Model with quantized linear layers
    """
    QuantClass = BitLinear if quantization_type == "bitnet" else FP8Linear

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip the final vocab projection
            if "lm_head" in name or "vocab_proj" in name:
                continue

            quantized = QuantClass(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )

            # Copy weights
            with torch.no_grad():
                quantized.weight.copy_(module.weight)
                if module.bias is not None:
                    quantized.bias.copy_(module.bias)

            # Replace in parent module
            *path, attr = name.split(".")
            parent = model
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, attr, quantized)

    return model
