"""
SAINT Agent Reasoning Engine.

Wraps the classifier with a structured decision layer that:
  1. Runs inference and extracts per-class confidence scores.
  2. Applies threshold logic to autonomously classify / flag / escalate.
  3. Produces a structured ThreatDecision with human-readable rationale.
  4. Maintains a short-term memory of recent decisions for pattern detection.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FLAG_FOR_REVIEW_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD,
    IDX_TO_CATEGORY, MODEL_NUM_CLASSES,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

SEVERITY = {
    "normal": 0,
    "probe":  1,
    "r2l":    2,
    "dos":    3,
    "u2r":    4,
}

@dataclass
class ThreatDecision:
    decision_id: str
    timestamp: float
    # Raw model outputs
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    # Agent reasoning outputs
    action: str          # "allow" | "flag" | "block"
    severity: int        # 0–4
    rationale: str
    # Optional: flagged for human review
    needs_review: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reasoning engine
# ---------------------------------------------------------------------------

class ThreatReasoningAgent:
    """
    Stateful agent that wraps the SAINT classifier and applies
    rule-based + statistical reasoning to produce actionable decisions.
    """

    def __init__(
        self,
        model,
        device: str | None = None,
        memory_size: int = 500,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Short-term memory: deque of recent ThreatDecisions
        self._memory: Deque[ThreatDecision] = deque(maxlen=memory_size)

        # Running counters for anomaly burst detection
        self._burst_window: Deque[str] = deque(maxlen=50)

    # ------------------------------------------------------------------
    # Core decision method
    # ------------------------------------------------------------------

    def decide(self, features: np.ndarray) -> ThreatDecision:
        """
        Given a preprocessed feature vector (1, D), return a ThreatDecision.
        """
        proba = self._infer(features)            # shape: (5,)
        pred_idx = int(np.argmax(proba))
        pred_class = IDX_TO_CATEGORY[pred_idx]
        confidence = float(proba[pred_idx])

        class_probabilities = {
            IDX_TO_CATEGORY[i]: float(proba[i]) for i in range(MODEL_NUM_CLASSES)
        }

        action, needs_review, rationale = self._apply_rules(
            pred_class, confidence, proba
        )

        decision = ThreatDecision(
            decision_id=str(uuid.uuid4()),
            timestamp=time.time(),
            predicted_class=pred_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            action=action,
            severity=SEVERITY.get(pred_class, 0),
            rationale=rationale,
            needs_review=needs_review,
        )

        self._update_memory(decision)
        return decision

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self, features: np.ndarray) -> np.ndarray:
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            proba = self.model.predict_proba(x)
        return proba.cpu().numpy().squeeze()

    # ------------------------------------------------------------------
    # Rule-based reasoning layer
    # ------------------------------------------------------------------

    def _apply_rules(
        self,
        pred_class: str,
        confidence: float,
        proba: np.ndarray,
    ) -> tuple[str, bool, str]:
        """
        Returns (action, needs_review, rationale).

        Decision logic:
        - normal + high confidence         → allow
        - normal + low confidence          → flag (borderline)
        - attack + high confidence         → block
        - attack + medium confidence       → flag for review
        - attack + low confidence          → flag for review
        - burst pattern detected           → escalate all to block
        """
        is_attack = pred_class != "normal"
        burst = self._detect_burst()

        if burst:
            action = "block"
            needs_review = False
            rationale = (
                f"Burst attack pattern detected in recent traffic. "
                f"Predicted: {pred_class} (conf={confidence:.2f}). "
                f"Escalating to block."
            )
        elif not is_attack and confidence >= HIGH_CONFIDENCE_THRESHOLD:
            action = "allow"
            needs_review = False
            rationale = (
                f"High-confidence normal traffic (conf={confidence:.2f}). "
                f"No threat indicators."
            )
        elif not is_attack and confidence < FLAG_FOR_REVIEW_THRESHOLD:
            action = "flag"
            needs_review = True
            rationale = (
                f"Low-confidence normal classification (conf={confidence:.2f}). "
                f"Secondary probabilities: {self._top2_str(proba)}. "
                f"Flagged for human review."
            )
        elif is_attack and confidence >= HIGH_CONFIDENCE_THRESHOLD:
            action = "block"
            needs_review = False
            rationale = (
                f"High-confidence {pred_class.upper()} attack detected "
                f"(conf={confidence:.2f}). Autonomously blocked."
            )
        else:
            action = "flag"
            needs_review = True
            rationale = (
                f"Possible {pred_class.upper()} attack with moderate confidence "
                f"(conf={confidence:.2f}). Requires human verification. "
                f"Top alternatives: {self._top2_str(proba)}"
            )

        return action, needs_review, rationale

    # ------------------------------------------------------------------
    # Pattern detection helpers
    # ------------------------------------------------------------------

    def _detect_burst(self) -> bool:
        """True if >60% of the last 50 decisions are attack traffic."""
        if len(self._burst_window) < 20:
            return False
        attack_ratio = sum(1 for c in self._burst_window if c != "normal") / len(
            self._burst_window
        )
        return attack_ratio > 0.60

    def _update_memory(self, decision: ThreatDecision) -> None:
        self._memory.append(decision)
        self._burst_window.append(decision.predicted_class)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _top2_str(proba: np.ndarray) -> str:
        top2_idx = np.argsort(proba)[-2:][::-1]
        return ", ".join(
            f"{IDX_TO_CATEGORY[i]}={proba[i]:.2f}" for i in top2_idx
        )

    def recent_decisions(self, n: int = 100) -> list[dict]:
        """Return the last n decisions as dicts (for dashboard/API)."""
        decisions = list(self._memory)[-n:]
        return [d.to_dict() for d in reversed(decisions)]

    def stats(self) -> dict:
        """Aggregate stats over the current memory window."""
        if not self._memory:
            return {}
        decisions = list(self._memory)
        total = len(decisions)
        action_counts = {"allow": 0, "flag": 0, "block": 0}
        class_counts: dict[str, int] = {}
        for d in decisions:
            action_counts[d.action] = action_counts.get(d.action, 0) + 1
            class_counts[d.predicted_class] = class_counts.get(d.predicted_class, 0) + 1
        return {
            "total_processed": total,
            "action_counts": action_counts,
            "class_counts": class_counts,
            "burst_detected": self._detect_burst(),
            "avg_confidence": float(np.mean([d.confidence for d in decisions])),
        }
