# @WHO:   self-ai/src/derivation/experience_metrics.py
# @WHAT:  Custom evaluation metrics for ExperienceWeight learning quality
# @PART:  self-ai/derivation
# @ENTRY: ExperienceMetrics.compute_all()

"""Experience Metrics — HOW WELL does SELF learn from mistakes?

These metrics measure LEARNING, not ACCURACY.
Standard NLP benchmarks measure accuracy (how many answers are correct).
We measure HOW WELL THE SYSTEM LEARNS FROM MISTAKES — a fundamentally
different question that no existing benchmark covers.

Why custom metrics:
    - MMLU measures knowledge breadth, not learning ability
    - Accuracy alone doesn't tell us if the system is improving
    - A system that gets 80% correct but never improves is worse than
      one that starts at 50% but learns rapidly
    - We need to measure: learning speed, stability, and whether
      experience adjustments actually help

Metric Design Philosophy:
    1. Learning Velocity — positive = learning, negative = regressing
    2. Correction Stability — low relapse rate is key
    3. Penalty Effectiveness — are adjustments helping or hurting?
    4. Threshold Health — is the adaptive threshold converging?
    5. Experience Coverage — is experience spreading across nodes?

Usage:
    metrics = ExperienceMetrics(store)
    report = metrics.compute_all()
    print(f"Learning velocity: {report['learning_velocity']:.3f}")
    print(f"Correction stability: {report['correction_stability']:.3f}")
"""

import time
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExperienceMetrics:
    """Compute custom metrics for ExperienceWeight evaluation.

    These metrics measure HOW WELL THE SYSTEM LEARNS FROM MISTAKES,
    not how accurate it is. This is a fundamentally different measurement
    from standard NLP benchmarks.

    Each metric returns a float in [0, 1] or [-1, 1]:
    - Learning Velocity: [-1, 1] — positive = improving, negative = regressing
    - Correction Stability: [0, 1] — high = few relapses
    - Penalty Effectiveness: [0, 1] — high = adjustments help
    - Threshold Health: [0, 1] — high = threshold is converging
    - Experience Coverage: [0, 1] — high = experience is widespread
    """

    def __init__(self, experience_store):
        """Initialize metrics with an ExperienceStore instance.

        Args:
            experience_store: ExperienceStore instance to compute metrics from
        """
        self._store = experience_store

    def compute_all(self, total_nodes: int = 0) -> dict:
        """Compute all metrics and return a report dict.

        @FLOW:     EXPERIENCE_METRICS
        @CALLS:    ExperienceStore.get_stats(), self._store internal state
        @MUTATES:  none — read-only computation
        @BEHAVIOR: Returns a dict with all metric values plus store stats.
                   Metrics that can't be computed (insufficient data) return
                   None instead of a float. Callers should check for None.

        Args:
            total_nodes: Total number of understanding nodes in the graph.
                         Needed for experience_coverage metric. If 0,
                         coverage metric returns None.

        Returns:
            dict with metric values and store statistics
        """
        stats = self._store.get_stats()

        report = {
            'learning_velocity': self.learning_velocity(),
            'correction_stability': self.correction_stability(),
            'penalty_effectiveness': self.penalty_effectiveness(),
            'threshold_health': self.threshold_health(),
            'experience_coverage': self.experience_coverage(total_nodes),
            'store_stats': stats,
        }

        logger.debug("Experience metrics: velocity=%.3f stability=%.3f "
                     "effectiveness=%.3f threshold_health=%.3f coverage=%.3f",
                     report['learning_velocity'] or 0,
                     report['correction_stability'] or 0,
                     report['penalty_effectiveness'] or 0,
                     report['threshold_health'] or 0,
                     report['experience_coverage'] or 0)

        return report

    def learning_velocity(self, window_hours: float = 24.0) -> Optional[float]:
        """How fast the failure rate decreases over time.

        Learning velocity measures the RATE OF CHANGE in failure ratio
        over the specified time window. A positive value means the system
        is learning (fewer failures over time), negative means it's
        regressing (more failures over time).

        Computation:
            1. Split episodes in the time window into early half and late half
            2. Compute failure_ratio for each half
            3. Velocity = (early_ratio - late_ratio)
            4. Normalize to [-1, 1] range

        Args:
            window_hours: Time window to look back (default 24 hours)

        Returns:
            Float in [-1, 1], or None if insufficient data
        """
        now = time.time()
        window_secs = window_hours * 3600
        cutoff = now - window_secs

        # Get episodes within window
        episodes = [
            ep for ep in self._store._episodes
            if ep.get('timestamp', 0) >= cutoff
        ]

        if len(episodes) < 4:
            return None  # Need at least 4 episodes for meaningful split

        # Split into early and late halves
        mid_time = (cutoff + now) / 2.0
        early = [ep for ep in episodes if ep.get('timestamp', 0) < mid_time]
        late = [ep for ep in episodes if ep.get('timestamp', 0) >= mid_time]

        if not early or not late:
            return None

        # Compute failure ratios
        early_failures = sum(1 for ep in early if ep.get('outcome') == 'failure')
        late_failures = sum(1 for ep in late if ep.get('outcome') == 'failure')

        early_ratio = early_failures / len(early)
        late_ratio = late_failures / len(late)

        # Velocity: positive = learning (fewer failures over time)
        # Normalize to [-1, 1] range
        velocity = early_ratio - late_ratio

        return round(velocity, 4)

    def correction_stability(self) -> Optional[float]:
        """How stable the correction is — low relapse rate.

        A relapse is when a failure follows a success for the same node,
        meaning the system "learned" something but then made the same
        mistake again. Low relapse = high stability.

        Computation:
            1. Sort episodes by timestamp
            2. For each node, count relapses (failure after success)
            3. stability = 1 - (relapse_count / max(1, success_count))

        Returns:
            Float in [0, 1], or None if insufficient data
        """
        match_history = self._store._match_history

        if len(match_history) < 3:
            return None

        # Sort by timestamp
        sorted_history = sorted(match_history, key=lambda x: x.get('timestamp', 0))

        # Count relapses: was_correct=True followed by was_correct=False
        relapses = 0
        successes = 0

        for i, entry in enumerate(sorted_history):
            if entry.get('was_correct', False):
                successes += 1
                # Check if next entry is a failure (relapse)
                if i + 1 < len(sorted_history):
                    next_entry = sorted_history[i + 1]
                    if not next_entry.get('was_correct', True):
                        relapses += 1

        if successes == 0:
            return None

        stability = 1.0 - (relapses / successes)
        stability = max(0.0, min(1.0, stability))

        return round(stability, 4)

    def penalty_effectiveness(self) -> Optional[float]:
        """How often experience adjustments lead to correct answers.

        This metric ONLY counts outcomes where an experience adjustment
        was actually applied (adjustment_applied=True). If adjustment
        was applied but answer was still wrong, that's a false positive.

        Computation:
            effective = count(was_correct AND adjustment_applied)
            total = count(adjustment_applied)
            effectiveness = effective / total

        Returns:
            Float in [0, 1], or None if no adjustments were applied
        """
        match_history = self._store._match_history

        # Filter to only entries where adjustment was applied
        adjusted = [
            e for e in match_history
            if e.get('adjustment_applied', False)
        ]

        if not adjusted:
            return None

        correct_after_adjustment = sum(
            1 for e in adjusted if e.get('was_correct', False)
        )

        effectiveness = correct_after_adjustment / len(adjusted)

        return round(effectiveness, 4)

    def threshold_health(self) -> Optional[float]:
        """Whether the adaptive threshold is converging or oscillating.

        A healthy threshold converges to a stable value (low std dev).
        An unhealthy threshold oscillates wildly (high std dev relative
        to its range).

        Computation:
            1. Get threshold history values
            2. Compute standard deviation
            3. health = 1 - (std_dev / range)
            4. Where range = THRESHOLD_MAX - THRESHOLD_MIN

        Returns:
            Float in [0, 1], or None if insufficient history
        """
        threshold_history = self._store._threshold_history

        if len(threshold_history) < 3:
            return None  # Need at least 3 data points

        values = [e['threshold'] for e in threshold_history]

        std_dev = float(np.std(values))
        threshold_range = self._store.THRESHOLD_MAX - self._store.THRESHOLD_MIN

        if threshold_range == 0:
            return None

        health = 1.0 - min(1.0, std_dev / threshold_range)

        return round(health, 4)

    def experience_coverage(self, total_nodes: int) -> Optional[float]:
        """How many nodes have experience data.

        Experience coverage measures how evenly experience is spread
        across understanding nodes. Low coverage means only a few nodes
        have experience, which could indicate the system only encounters
        certain types of questions.

        Computation:
            nodes_with_episodes = count of unique node_ids in episodes
            coverage = nodes_with_episodes / total_nodes

        Args:
            total_nodes: Total number of understanding nodes in the graph.
                         If 0, returns None.

        Returns:
            Float in [0, 1], or None if total_nodes is 0
        """
        if total_nodes <= 0:
            return None

        stats = self._store.get_stats()
        nodes_with_episodes = stats.get('unique_nodes', 0)

        coverage = nodes_with_episodes / total_nodes
        coverage = min(1.0, coverage)  # Cap at 1.0

        return round(coverage, 4)
