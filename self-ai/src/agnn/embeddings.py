"""AGNN Embeddings — model-native embedding extraction.

This module extracts embeddings directly from a language model's internal
representations, eliminating the dependency on external embedding models
like bge-m3. By reading hidden states from the model's own layers, we
obtain embeddings that are natively aligned with the model's semantic
space — no projection matrix needed.

Approach:
    - Extract hidden states from a specified layer (e.g., the last layer
      or a middle layer) of a HuggingFace transformer model.
    - Use the hidden state at the [CLS] token or mean-pool across all
      tokens as the node embedding.
    - These embeddings are guaranteed to be in the same vector space as
      the model's own representations, so message passing outputs can
      directly influence generation without any projection.

Advantages over bge-m3:
    - No external model dependency (smaller footprint)
    - Embeddings are naturally aligned with the LM's semantic space
    - No projection matrix training needed
    - Works with any HuggingFace transformer model

Status: Placeholder — no implementation yet.
"""

# TODO: Define ModelNativeEmbedder, extraction functions
