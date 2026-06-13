"""
AAM Validation Gates — 5 Pillar Foundation

Five meta-principles derived from quantitative trading, applied as
validation gates at each cognitive layer. Data that fails a gate is
REJECTED — structurally preventing hallucination, overconfidence,
and ungrounded reasoning.

The 5 Pillars:
  1. Signal Extraction  — Layer 0/1 gate: filter signal from noise
  2. Regime Detection   — Layer 2 gate: detect current cognitive environment
  3. Uncertainty Calibration — Layer 3 gate: calibrate confidence vs reality
  4. Statistical Edge   — Layer 4 gate: validate reasoning has positive EV
  5. Execution Discipline — Layer 5 gate: enforce output rules

Pipeline:
    Raw Input
      -> [GATE 1: Signal Extraction] -> Layer 0 Abstraction -> Layer 1 RSVS
      -> [GATE 2: Regime Detection] -> Layer 2 Situation/Context
      -> [GATE 3: Uncertainty Calibration] -> Layer 3 Reasoning
      -> [GATE 4: Statistical Edge] -> Layer 4 Predictive Coding
      -> [GATE 5: Execution Discipline] -> Layer 5 Output

Without these gates:
  AI cuma lihat "chart goes brrrr" — semua data dianggap sama,
  semua confidence dianggap benar, semua reasoning dianggap valid.

With these gates:
  Setiap layer punya validation checkpoint. Data yang tidak lolos = DITOLAK.
  Ini yang bikin AAM bukan cuma "AI yang bisa ngomong",
  tapi SISTEM REASONING YANG TERBUKTI.

Philosophy:
  "chatbot trader != quant system"
  "language model != validated reasoning system"
  AAM = the quant system of AI.
"""

from .signal_extraction import SignalExtractionGate, SignalResult, SignalVerdict
from .regime_detection import RegimeDetectionGate, RegimeState
from .uncertainty_calibration import UncertaintyCalibrationGate, CalibrationRecord
from .statistical_edge import StatisticalEdgeGate, EdgeAssessment, ReasoningPath
from .execution_discipline import ExecutionDisciplineGate, DisciplineVerdict

__all__ = [
    "SignalExtractionGate", "SignalResult", "SignalVerdict",
    "RegimeDetectionGate", "RegimeState",
    "UncertaintyCalibrationGate", "CalibrationRecord",
    "StatisticalEdgeGate", "EdgeAssessment", "ReasoningPath",
    "ExecutionDisciplineGate", "DisciplineVerdict",
]
