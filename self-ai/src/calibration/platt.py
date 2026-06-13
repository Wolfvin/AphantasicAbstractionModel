# @WHO:   self-ai/src/calibration/platt.py
# @WHAT:  Confidence Calibration — Platt Scaling
# @PART:  self-ai/calibration
# @ENTRY: PlattScaler

"""Confidence Calibration — Platt Scaling
Raw confidence scores from the system are not calibrated.
Platt Scaling fits: calibrated = sigmoid(a * raw + b)
using logistic regression on correctness labels.
"""
import numpy as np
from typing import List, Tuple, Optional
import json
import os


class PlattScaler:
    def __init__(self, cache_dir=None):
        self.a = 1.0
        self.b = 0.0
        self.fitted = False
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'calibration'
        )
        self.history = []  # (raw_confidence, is_correct) pairs

    def add_observation(self, raw_confidence: float, is_correct: bool):
        """Add a calibration observation"""
        self.history.append((raw_confidence, is_correct))

    def fit(self, observations: List[Tuple[float, bool]] = None):
        """Fit Platt Scaling parameters using logistic regression
        calibrated = sigmoid(a * raw + b)
        """
        if observations is None:
            observations = self.history

        if len(observations) < 10:
            # Not enough data to fit — use identity
            self.a = 1.0
            self.b = 0.0
            self.fitted = False
            return

        # Simple gradient descent for logistic regression
        X = np.array([obs[0] for obs in observations])
        y = np.array([1.0 if obs[1] else 0.0 for obs in observations])

        a, b = 1.0, 0.0
        lr = 0.01
        n_epochs = 1000

        for _ in range(n_epochs):
            z = a * X + b
            z = np.clip(z, -500, 500)  # Prevent overflow
            pred = 1.0 / (1.0 + np.exp(-z))

            # Gradient
            error = pred - y
            da = np.mean(error * X)
            db = np.mean(error)

            a -= lr * da
            b -= lr * db

        self.a = a
        self.b = b
        self.fitted = True

    def calibrate(self, raw_confidence: float) -> float:
        """Calibrate a raw confidence score"""
        z = self.a * raw_confidence + self.b
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def expected_calibration_error(self, observations: List[Tuple[float, bool]] = None, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)
        ECE = sum over bins of |avg_confidence - avg_accuracy| * bin_weight
        """
        if observations is None:
            observations = self.history

        if not observations:
            return 0.0

        X = np.array([obs[0] for obs in observations])
        y = np.array([1.0 if obs[1] else 0.0 for obs in observations])

        # Calibrate
        calibrated = np.array([self.calibrate(x) for x in X])

        # Bin
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (calibrated >= bin_boundaries[i]) & (calibrated < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            avg_conf = calibrated[mask].mean()
            avg_acc = y[mask].mean()
            weight = mask.sum() / len(y)
            ece += abs(avg_conf - avg_acc) * weight

        return ece

    def save(self):
        """Save calibration parameters"""
        os.makedirs(self.cache_dir, exist_ok=True)
        data = {
            'a': float(self.a),
            'b': float(self.b),
            'fitted': self.fitted,
            'history': self.history[-1000:],  # Keep last 1000
        }
        filepath = os.path.join(self.cache_dir, 'platt_params.json')
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self) -> bool:
        """Load calibration parameters"""
        filepath = os.path.join(self.cache_dir, 'platt_params.json')
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath) as f:
                data = json.load(f)
            self.a = data.get('a', 1.0)
            self.b = data.get('b', 0.0)
            self.fitted = data.get('fitted', False)
            self.history = [tuple(obs) for obs in data.get('history', [])]
            return True
        except Exception:
            return False
