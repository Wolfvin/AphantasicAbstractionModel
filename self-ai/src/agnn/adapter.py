"""AGNN Adapter — plug into any HuggingFace transformer.

This module provides a portable adapter that automatically detects a
model's configuration (hidden_size, num_layers, etc.) and wires up
the AGNN message passing pipeline accordingly. The goal is zero-config
integration: given any HuggingFace AutoModelForCausalLM, the adapter
should be able to:
    1. Read model.config to determine hidden_size, num_layers, etc.
    2. Initialize AGNN components with the correct dimensions.
    3. Hook into the model's forward pass to inject aggregated graph
       information into the appropriate hidden states.
    4. Provide a clean interface for the rest of the system.

This replaces the previous UnconsciousInjector approach. Instead of
a hardcoded projection from bge-m3 space, the adapter dynamically
configures itself based on the target model's architecture.

Status: Placeholder — no implementation yet.
"""

# TODO: define AGNNAdapter class
