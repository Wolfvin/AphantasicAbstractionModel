# @WHO:   self-ai/config/thresholds.py
# @WHAT:  Adaptive thresholds that respond to real feedback
# @PART:  self-ai/config
# @ENTRY: AdaptiveThresholds

"""Adaptive Thresholds - Respond to feedback
Thresholds adjust based on real feedback from the system.

Also contains module-level constants used across SELF-AI layers.
These constants are infrastructure defaults, NOT domain knowledge.
"""
import os
import json

# ── Layer-specific threshold constants ──
# These are structural/infrastructure defaults, not hardcoded domain knowledge.
# They control HOW the pipeline processes, not WHAT it knows.

# Translation layer
IDENTITY_THRESHOLD = 0.85       # Cosine similarity to consider nodes identical
MINIMUM_NODES = 3               # Minimum nodes before identity merge

# Memory layer
GLOBAL_LENS_WEIGHT = 0.3        # Weight for global context in memory retrieval
MEMORY_TOP_K = 5                # Number of axioms to retrieve from memory

# Concept layer
MIN_PATTERN_REPEAT = 3          # Observations needed before promoting to concept
CONCEPT_CONFIDENCE_INIT = 0.3   # Starting confidence for new concepts

# Axiom layer
AXIOM_MIN_CONFIDENCE = 0.6      # Minimum confidence to store an axiom
LENS_WEIGHT_DEFAULT = 0.5       # Default weight for lens filtering

# Curiosity engine
CURIOSITY_WEIGHT_INCONSISTENCY = 0.4
CURIOSITY_WEIGHT_UNEXPLAINED = 0.3
CURIOSITY_WEIGHT_FAILED_DERIVATION = 0.3


class _AdaptiveModule:
    """Module-level adaptive interface for curiosity weights."""

    def get_curiosity_weights(self) -> dict:
        """Get current curiosity weights."""
        at = AdaptiveThresholds.get_instance()
        return {
            "inconsistency": CURIOSITY_WEIGHT_INCONSISTENCY,
            "unexplained": CURIOSITY_WEIGHT_UNEXPLAINED,
            "derivation": CURIOSITY_WEIGHT_FAILED_DERIVATION,
        }

# Module-level adaptive instance
adaptive = _AdaptiveModule()


class AdaptiveThresholds:
    _instance = None

    def __init__(self):
        self.confidence_threshold = 0.5
        self.novelty_threshold = 0.7
        self.consistency_threshold = 0.8
        self.promotion_threshold = 3
        self.feedback_count = 0
        self.correct_count = 0
        self.recent_accuracy = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def update_from_feedback(self, is_correct: bool):
        """Update thresholds based on feedback"""
        self.feedback_count += 1
        if is_correct:
            self.correct_count += 1

        self.recent_accuracy.append(1.0 if is_correct else 0.0)
        # Keep last 20 feedback items
        self.recent_accuracy = self.recent_accuracy[-20:]

        avg_accuracy = sum(self.recent_accuracy) / len(self.recent_accuracy) if self.recent_accuracy else 0.5

        # Adjust thresholds based on accuracy
        if avg_accuracy > 0.8:
            # High accuracy - we can be more confident, raise thresholds slightly
            self.confidence_threshold = min(0.8, self.confidence_threshold + 0.01)
            self.promotion_threshold = max(2, self.promotion_threshold - 1)
        elif avg_accuracy < 0.5:
            # Low accuracy - be more cautious, lower thresholds
            self.confidence_threshold = max(0.3, self.confidence_threshold - 0.01)
            self.promotion_threshold = min(5, self.promotion_threshold + 1)

    def should_accept(self, confidence: float) -> bool:
        """Check if a result should be accepted"""
        return confidence >= self.confidence_threshold

    def should_promote(self, observation_count: int) -> bool:
        """Check if a rule should be promoted"""
        return observation_count >= self.promotion_threshold

    def save(self, filepath: str = None):
        """Save thresholds to file"""
        filepath = filepath or os.path.join(os.path.dirname(__file__), '..', 'data', 'thresholds.json')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            'confidence_threshold': self.confidence_threshold,
            'novelty_threshold': self.novelty_threshold,
            'consistency_threshold': self.consistency_threshold,
            'promotion_threshold': self.promotion_threshold,
            'feedback_count': self.feedback_count,
            'correct_count': self.correct_count
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: str = None):
        """Load thresholds from file"""
        filepath = filepath or os.path.join(os.path.dirname(__file__), '..', 'data', 'thresholds.json')
        if os.path.exists(filepath):
            with open(filepath) as f:
                state = json.load(f)
            self.confidence_threshold = state.get('confidence_threshold', 0.5)
            self.novelty_threshold = state.get('novelty_threshold', 0.7)
            self.consistency_threshold = state.get('consistency_threshold', 0.8)
            self.promotion_threshold = state.get('promotion_threshold', 3)
            self.feedback_count = state.get('feedback_count', 0)
            self.correct_count = state.get('correct_count', 0)
