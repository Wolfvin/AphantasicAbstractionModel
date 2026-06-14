# @WHO:   self-ai/src/unconscious/projection_trainer.py
# @WHAT:  Train the projection layer in UnconsciousInjector for directional alignment
# @PART:  self-ai/unconscious
# @ENTRY: ProjectionTrainer

"""ProjectionTrainer — train UnconsciousInjector's projection for directional alignment.

Problem:
    UnconsciousInjector projects bge-m3 embeddings (1024-dim) into Qwen3's
    hidden state space via a Linear projection. This projection is initialized
    as identity and NOT trained — so injected vectors don't actually point
    toward the hidden states that produce "correct" output.

    Benchmark shows directional alignment = -0.02 (essentially random).
    The injection changes output, but not in the *right direction*.

Solution:
    Train the projection so that experience vectors map toward the hidden
    states Qwen3 produces when processing the "correct" output associated
    with that experience.

    Training pipeline:
      1. For each (experience_text, correct_output_text) pair:
         - Encode experience_text via bge-m3 → experience_vector (1024-dim)
         - Forward pass Qwen3 with correct_output_text → hidden state at
           target layer (layer 14), last token position
      2. Loss: cosine similarity between projection(experience_vector) and
         hidden_state_target
      3. Optimize projection weights via Adam on CPU
      4. Save trained weights to projection_weights.pt

Architecture:
    experience_text → bge-m3 → experience_vector (1024-dim)
                                    ↓
                            projection (Linear 1024→1024)
                                    ↓
                            projected_vector (1024-dim)
                                    ↓
                        cosine similarity loss ← hidden_state_target
                                                    ↑
    correct_output_text → Qwen3 forward → hook at layer 14, last token

Usage:
    from unconscious.projection_trainer import ProjectionTrainer
    from unconscious.injector import UnconsciousInjector

    # Initialize with pre-loaded models
    trainer = ProjectionTrainer(qwen3_model, qwen3_tokenizer, embedding_model=bge_model)

    # Train on experience-output pairs
    pairs = [
        ("Ibu kota Indonesia adalah Jakarta", "Jakarta"),
        ("2 + 2 sama dengan", "4"),
        ("Air mendidih pada suhu", "100 derajat Celcius"),
    ]
    trainer.train(pairs, epochs=50)

    # Load trained weights into an injector
    injector = UnconsciousInjector(qwen3_model)
    trainer.load_into_injector(injector)

    # Now injector._projection uses trained weights instead of identity init

Constraint:
    - Projection is always torch.nn.Linear(1024, 1024, bias=False)
    - Training is CPU-only (no GPU required)
    - Does NOT change UnconsciousInjector's public interface
    - Does NOT hardcode model paths — accepts models in constructor
"""

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Default path for saving/loading projection weights — same directory as this file
_PROJECTION_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'projection_weights.pt'
)


class ProjectionTrainer:
    """Train the projection layer in UnconsciousInjector for directional alignment.

    The projection maps experience vectors (bge-m3, 1024-dim) to Qwen3 hidden
    state space (1024-dim for Qwen3-0.6B) so that injection steers the model
    toward hidden states associated with "correct" output.

    Attributes:
        model: The Qwen3 model (AutoModelForCausalLM instance).
        tokenizer: Qwen3 tokenizer.
        embedding_model: bge-m3 SentenceTransformer instance (optional).
        hook_layer_index: Which transformer layer to capture hidden states from.
        learning_rate: Learning rate for Adam optimizer.
        projection: The trainable Linear(1024, 1024, bias=False) projection.
    """

    BGE_EMBEDDING_DIM = 1024

    def __init__(self, model, tokenizer, embedding_model=None,
                 hook_layer_index: int = 14, learning_rate: float = 1e-3):
        """Initialize the ProjectionTrainer.

        Args:
            model: A Qwen3 model (AutoModelForCausalLM instance).
                Must have model.model.layers attribute.
            tokenizer: Qwen3 tokenizer for encoding text to input_ids.
            embedding_model: bge-m3 SentenceTransformer instance.
                If None, will attempt to load via model_registry.
            hook_layer_index: Which transformer layer to capture hidden
                states from (default 14, middle of 28 layers).
            learning_rate: Learning rate for Adam optimizer (default 1e-3).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.hook_layer_index = hook_layer_index
        self.learning_rate = learning_rate

        # Read hidden size from model config — don't hardcode
        self.hidden_size = model.config.hidden_size

        # The trainable projection: Linear(1024, hidden_size, bias=False)
        # For Qwen3-0.6B: Linear(1024, 1024, bias=False)
        self.projection = torch.nn.Linear(
            self.BGE_EMBEDDING_DIM, self.hidden_size, bias=False
        )

        # Initialize as identity (same as UnconsciousInjector default)
        self._init_projection_identity()

        # Ensure model is in eval mode (we don't train Qwen3)
        self.model.eval()

        # Try to get embedding model from registry if not provided
        if self.embedding_model is None:
            self._try_load_embedding_model()

        logger.info(
            "ProjectionTrainer initialized: projection=%s, layer=%d, lr=%.1e",
            f"Linear({self.BGE_EMBEDDING_DIM}, {self.hidden_size})",
            self.hook_layer_index,
            self.learning_rate,
        )

    def _init_projection_identity(self):
        """Initialize projection weights as identity matrix.

        For the square case (1024→1024), this is a pure identity matrix.
        For the non-square case, the overlapping dimensions get identity
        and the rest get small random values.
        """
        with torch.no_grad():
            n_out = self.hidden_size
            n_in = self.BGE_EMBEDDING_DIM
            weight = torch.zeros(n_out, n_in)

            overlap = min(n_out, n_in)
            weight[:overlap, :overlap] = torch.eye(overlap)

            # Extra columns (embedding dim > hidden size)
            if n_in > n_out:
                torch.nn.init.xavier_uniform_(weight[:, n_out:])
                weight[:, n_out:] *= 0.1

            self.projection.weight.copy_(weight)

    def _try_load_embedding_model(self):
        """Try to load bge-m3 from the model registry singleton."""
        try:
            from derivation.model_registry import get_shared_embedding_model
            self.embedding_model = get_shared_embedding_model()
            if self.embedding_model is not None:
                logger.info("ProjectionTrainer: using shared bge-m3 from model_registry")
            else:
                logger.warning(
                    "ProjectionTrainer: bge-m3 not available from registry. "
                    "Pass embedding_model explicitly to train()."
                )
        except ImportError:
            logger.warning(
                "ProjectionTrainer: model_registry not available. "
                "Pass embedding_model explicitly to train()."
            )

    def _encode_experience(self, text: str) -> Optional[torch.Tensor]:
        """Encode experience text to a 1024-dim vector via bge-m3.

        Args:
            text: The experience text to encode.

        Returns:
            Normalized 1024-dim float32 tensor, or None if encoding fails.
        """
        if self.embedding_model is None:
            logger.error("No embedding model available — cannot encode experience text")
            return None

        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return torch.tensor(embedding, dtype=torch.float32)
        except Exception as e:
            logger.error("Failed to encode experience text '%s...': %s", text[:50], e)
            return None

    def _capture_hidden_state(self, text: str) -> Optional[torch.Tensor]:
        """Forward pass Qwen3 and capture hidden state at target layer, last token.

        Uses a forward hook on the target transformer layer to capture
        the hidden state at the last token position. The model is kept
        in eval mode and no gradients are computed for Qwen3 parameters.

        Args:
            text: The correct output text to process through Qwen3.

        Returns:
            Float32 tensor of shape (hidden_size,) — the hidden state at
            the target layer for the last token position. Returns None
            if capture fails.
        """
        captured = {}

        def _hook_fn(module, input, output):
            """Capture hook — store hidden state at last token position."""
            try:
                if isinstance(output, tuple):
                    hidden_states = output[0]
                elif isinstance(output, torch.Tensor):
                    hidden_states = output
                else:
                    return

                # Capture last token, convert to float32 for training
                captured['hidden'] = hidden_states[:, -1, :].detach().float().cpu()
            except Exception as e:
                logger.warning("Hook capture error: %s", e)

        # Register hook on the target layer
        try:
            target_layer = self.model.model.layers[self.hook_layer_index]
        except (AttributeError, IndexError) as e:
            logger.error("Cannot access Qwen3 layer %d: %s", self.hook_layer_index, e)
            return None

        handle = target_layer.register_forward_hook(_hook_fn)

        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text, return_tensors='pt', truncation=True, max_length=512
            )

            # Move input to model's device
            try:
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except (StopIteration, AttributeError):
                pass

            # Forward pass (no gradient for Qwen3)
            with torch.no_grad():
                self.model(**inputs)

            # Retrieve captured hidden state
            hidden = captured.get('hidden')
            if hidden is None:
                logger.warning("Hook did not capture hidden state for text '%s...'", text[:50])
                return None

            # Squeeze batch dimension → (hidden_size,)
            return hidden.squeeze(0)

        except Exception as e:
            logger.error("Failed to capture hidden state for '%s...': %s", text[:50], e)
            return None
        finally:
            handle.remove()

    def _prepare_training_data(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Prepare (experience_vector, hidden_state_target) pairs for training.

        Encodes experience texts via bge-m3 and captures Qwen3 hidden states
        for correct output texts. This is a one-time computation — the results
        are reused across all training epochs.

        Args:
            pairs: List of (experience_text, correct_output_text) tuples.

        Returns:
            List of (experience_vector, hidden_state_target) tensor tuples.
            Each tensor is float32 on CPU. Returns empty list if preparation
            fails for all pairs.
        """
        training_data = []

        logger.info("Preparing training data for %d pairs...", len(pairs))

        for i, (exp_text, correct_text) in enumerate(pairs):
            # Encode experience text → experience_vector (1024-dim)
            exp_vec = self._encode_experience(exp_text)
            if exp_vec is None:
                logger.warning("Skipping pair %d: failed to encode experience text", i)
                continue

            # Forward pass Qwen3 → hidden_state_target (hidden_size-dim)
            hidden_target = self._capture_hidden_state(correct_text)
            if hidden_target is None:
                logger.warning("Skipping pair %d: failed to capture hidden state", i)
                continue

            training_data.append((exp_vec, hidden_target))

            if (i + 1) % 5 == 0 or (i + 1) == len(pairs):
                logger.info("Prepared %d/%d pairs", i + 1, len(pairs))

        logger.info(
            "Training data ready: %d/%d pairs successfully prepared",
            len(training_data), len(pairs)
        )

        return training_data

    def train(self, pairs: List[Tuple[str, str]], epochs: int = 50) -> dict:
        """Train the projection on (experience_text, correct_output_text) pairs.

        The training process:
          1. Pre-compute (experience_vector, hidden_state_target) pairs
          2. For each epoch, compute projection(experience_vector) for all pairs
          3. Loss = 1 - cosine_similarity(projected, target)
          4. Backpropagate through projection weights only (Qwen3 is frozen)
          5. Save trained weights to projection_weights.pt

        Args:
            pairs: List of (experience_text, correct_output_text) tuples.
                experience_text: text describing the experience/context
                    (encoded via bge-m3 to get the experience vector).
                correct_output_text: the "correct" output text that Qwen3
                    should produce (forward-passed through Qwen3 to get the
                    target hidden state).
            epochs: Number of training epochs (default 50).

        Returns:
            Dict with training metrics:
                - 'final_loss': Average loss in the last epoch
                - 'initial_loss': Average loss in the first epoch
                - 'epochs_trained': Number of epochs actually trained
                - 'num_pairs': Number of training pairs used
                - 'weights_path': Path where weights were saved
                - 'directional_alignment': Final cosine similarity (avg)

        Raises:
            ValueError: If no valid training pairs can be prepared.
        """
        if not pairs:
            raise ValueError("No training pairs provided")

        # Step 1: Prepare training data (one-time compute)
        training_data = self._prepare_training_data(pairs)
        if not training_data:
            raise ValueError(
                "Could not prepare any training data — check that models "
                "are loaded and pairs are valid"
            )

        # Step 2: Set up training
        self.projection.train()
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=self.learning_rate)

        initial_loss = None
        final_loss = None

        logger.info(
            "Starting training: %d pairs, %d epochs, lr=%.1e",
            len(training_data), epochs, self.learning_rate
        )

        # Step 3: Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_cosine_sim = 0.0

            for exp_vec, hidden_target in training_data:
                optimizer.zero_grad()

                # Forward: project experience vector to hidden state space
                projected = self.projection(exp_vec)

                # Normalize both vectors for stable cosine similarity
                proj_norm = F.normalize(projected, dim=0)
                target_norm = F.normalize(hidden_target, dim=0)

                # Cosine similarity loss: maximize similarity = minimize (1 - cos_sim)
                cos_sim = F.cosine_similarity(
                    proj_norm.unsqueeze(0), target_norm.unsqueeze(0)
                ).squeeze()
                loss = 1.0 - cos_sim

                # Backward + update
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cosine_sim += cos_sim.item()

            avg_loss = epoch_loss / len(training_data)
            avg_cos_sim = epoch_cosine_sim / len(training_data)

            if initial_loss is None:
                initial_loss = avg_loss

            final_loss = avg_loss

            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                logger.info(
                    "Epoch %3d/%d: loss=%.4f, cosine_sim=%.4f",
                    epoch + 1, epochs, avg_loss, avg_cos_sim
                )

        # Step 4: Save trained weights
        self.projection.eval()
        weights_path = self.save_weights()

        metrics = {
            'final_loss': final_loss,
            'initial_loss': initial_loss,
            'epochs_trained': epochs,
            'num_pairs': len(training_data),
            'weights_path': weights_path,
            'directional_alignment': avg_cos_sim,
        }

        logger.info(
            "Training complete: initial_loss=%.4f → final_loss=%.4f, "
            "directional_alignment=%.4f, weights saved to %s",
            initial_loss, final_loss, avg_cos_sim, weights_path
        )

        return metrics

    def save_weights(self, path: Optional[str] = None) -> str:
        """Save projection weights to a .pt file.

        Args:
            path: File path to save weights. If None, uses default path
                (self-ai/src/unconscious/projection_weights.pt).

        Returns:
            The path where weights were saved.
        """
        save_path = path or _PROJECTION_WEIGHTS_PATH

        # Save the projection weight tensor (not the whole module)
        torch.save({
            'weight': self.projection.weight.data.cpu(),
            'config': {
                'in_features': self.projection.in_features,
                'out_features': self.projection.out_features,
                'bias': False,
                'hidden_size': self.hidden_size,
                'embedding_dim': self.BGE_EMBEDDING_DIM,
                'hook_layer_index': self.hook_layer_index,
            }
        }, save_path)

        logger.info("Projection weights saved to %s", save_path)
        return save_path

    def load_weights(self, path: Optional[str] = None) -> bool:
        """Load projection weights from a .pt file.

        Args:
            path: File path to load weights from. If None, uses default path.

        Returns:
            True if weights were loaded successfully, False otherwise.
        """
        load_path = path or _PROJECTION_WEIGHTS_PATH

        if not os.path.exists(load_path):
            logger.warning("Projection weights file not found: %s", load_path)
            return False

        try:
            checkpoint = torch.load(load_path, map_location='cpu', weights_only=True)

            # Validate dimensions match
            saved_config = checkpoint.get('config', {})
            if saved_config.get('in_features') != self.projection.in_features:
                logger.error(
                    "Weight dimension mismatch: saved in_features=%s, expected=%s",
                    saved_config.get('in_features'), self.projection.in_features
                )
                return False
            if saved_config.get('out_features') != self.projection.out_features:
                logger.error(
                    "Weight dimension mismatch: saved out_features=%s, expected=%s",
                    saved_config.get('out_features'), self.projection.out_features
                )
                return False

            self.projection.weight.data.copy_(checkpoint['weight'])
            logger.info("Projection weights loaded from %s", load_path)
            return True

        except Exception as e:
            logger.error("Failed to load projection weights from %s: %s", load_path, e)
            return False

    def load_into_injector(self, injector) -> bool:
        """Load trained projection weights into an UnconsciousInjector.

        This replaces the injector's default (identity-init) projection with
        the trained weights from this trainer. The injector's projection must
        have the same dimensions.

        Args:
            injector: An UnconsciousInjector instance.

        Returns:
            True if weights were loaded successfully, False otherwise.
        """
        try:
            injector_proj = injector._projection

            # Validate dimensions
            if injector_proj.in_features != self.projection.in_features:
                logger.error(
                    "Dimension mismatch: injector projection in_features=%s, "
                    "trainer projection in_features=%s",
                    injector_proj.in_features, self.projection.in_features
                )
                return False
            if injector_proj.out_features != self.projection.out_features:
                logger.error(
                    "Dimension mismatch: injector projection out_features=%s, "
                    "trainer projection out_features=%s",
                    injector_proj.out_features, self.projection.out_features
                )
                return False

            # Copy weights to injector's projection
            with torch.no_grad():
                injector_proj.weight.data.copy_(
                    self.projection.weight.data.cpu()
                )

            # Move to injector's model device
            try:
                device = next(injector.model.parameters()).device
                injector_proj.to(device)
            except (StopIteration, AttributeError):
                pass

            logger.info(
                "Trained projection weights loaded into UnconsciousInjector "
                "(shape=%s, device=%s)",
                tuple(injector_proj.weight.shape),
                injector_proj.weight.device,
            )
            return True

        except AttributeError as e:
            logger.error("Invalid injector object: %s", e)
            return False
