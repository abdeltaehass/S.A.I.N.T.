"""
LLM-powered SOC incident reporting for SAINT.

Turns a structured ThreatDecision into a natural-language analyst report using a
local, free LLM served by Ollama (https://ollama.com). The model never touches
the fast inference path — reports are generated on demand (API/dashboard).

Design goals:
  • Zero-cost & offline   — uses a local Ollama model, no API keys, no cloud.
  • Never breaks the app  — if Ollama is unreachable, a deterministic template
                            report is returned instead, so the feature degrades
                            gracefully and "upgrades" once Ollama is running.
  • No new dependencies   — talks to Ollama's REST API with `requests`.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    LLM_ENABLED, LLM_TIMEOUT, OLLAMA_HOST, OLLAMA_MODEL,
)

# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior SOC (Security Operations Center) analyst writing a concise "
    "incident report for a network intrusion detection alert. Be factual, calm, "
    "and specific. Do not invent details that are not in the alert data. Use the "
    "exact section headings requested and keep the whole report under 180 words."
)

_REPORT_TEMPLATE = """\
A network intrusion detection alert was raised. Write an incident report using \
EXACTLY these four sections (markdown headings):

### Summary
### Severity & Action
### Key Indicators
### Recommended Next Steps

Alert data:
- Predicted threat class: {predicted_class} ({class_desc})
- Model confidence: {confidence:.0%}
- Agent action taken: {action}
- Severity (0=info .. 4=critical): {severity}
- Needs human review: {needs_review}
- Class probability breakdown: {proba_str}
- Top contributing features (from Integrated Gradients): {drivers_str}
- Agent rationale: {rationale}

Write the report now."""

# Human-readable descriptions of the 5 NSL-KDD categories, for prompt context.
_CLASS_DESC = {
    "normal": "benign traffic",
    "dos":    "Denial-of-Service attack",
    "probe":  "reconnaissance / network scanning",
    "r2l":    "Remote-to-Local unauthorized access",
    "u2r":    "User-to-Root privilege escalation",
}


def _drivers_str(top_features: list[dict] | None) -> str:
    if not top_features:
        return "none recorded"
    parts = []
    for f in top_features[:5]:
        arrow = "elevated" if f.get("contribution", 0) > 0 else "suppressed"
        parts.append(f"{f.get('name', '?')} ({arrow})")
    return ", ".join(parts)


def _proba_str(class_probabilities: dict[str, float] | None) -> str:
    if not class_probabilities:
        return "n/a"
    ranked = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{k}={v:.0%}" for k, v in ranked if v >= 0.01)


def _build_prompt(decision: dict) -> str:
    pred = decision.get("predicted_class", "unknown")
    return _REPORT_TEMPLATE.format(
        predicted_class=pred,
        class_desc=_CLASS_DESC.get(pred, "unknown category"),
        confidence=decision.get("confidence", 0.0),
        action=decision.get("action", "flag"),
        severity=decision.get("severity", 0),
        needs_review="yes" if decision.get("needs_review") else "no",
        proba_str=_proba_str(decision.get("class_probabilities")),
        drivers_str=_drivers_str(decision.get("top_features")),
        rationale=decision.get("rationale", "n/a"),
    )


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class ThreatReporter:
    """
    Generates incident reports for ThreatDecisions.

    Call `generate(decision_dict)` to get:
        {"report": <markdown>, "source": "llm"|"template", "model": <name>,
         "generated_at": <epoch>, "elapsed_ms": <float>}
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        timeout: int = LLM_TIMEOUT,
        enabled: bool = LLM_ENABLED,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.enabled = enabled
        # Availability is cached briefly so we don't probe Ollama on every call.
        self._available: bool | None = None
        self._checked_at: float = 0.0

    # ------------------------------------------------------------------
    # Availability probe
    # ------------------------------------------------------------------

    def available(self, ttl: float = 30.0) -> bool:
        """True if Ollama is reachable. Result cached for `ttl` seconds."""
        if not self.enabled:
            return False
        now = time.time()
        if self._available is not None and (now - self._checked_at) < ttl:
            return self._available
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=2)
            self._available = resp.status_code == 200
        except requests.RequestException:
            self._available = False
        self._checked_at = now
        return self._available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, decision: dict) -> dict:
        """Produce an incident report, preferring the LLM, falling back to template."""
        t0 = time.perf_counter()
        if self.available():
            report = self._generate_llm(decision)
            if report:
                return {
                    "report": report,
                    "source": "llm",
                    "model": self.model,
                    "generated_at": time.time(),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
        # Fallback — deterministic, instant, always available.
        return {
            "report": self._generate_template(decision),
            "source": "template",
            "model": None,
            "generated_at": time.time(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ------------------------------------------------------------------
    # LLM backend (Ollama)
    # ------------------------------------------------------------------

    def _generate_llm(self, decision: dict) -> str | None:
        """Call Ollama /api/generate. Returns None on any failure (caller falls back)."""
        payload = {
            "model": self.model,
            "system": _SYSTEM_PROMPT,
            "prompt": _build_prompt(decision),
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 320},
        }
        try:
            resp = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            return text or None
        except (requests.RequestException, ValueError) as e:
            print(f"[llm_reporter] Ollama generation failed, using template: {e}")
            self._available = False     # force a re-probe next time
            return None

    # ------------------------------------------------------------------
    # Template backend (always available)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_template(decision: dict) -> str:
        """Deterministic markdown report built directly from the decision fields."""
        pred = decision.get("predicted_class", "unknown")
        desc = _CLASS_DESC.get(pred, "unknown category")
        conf = decision.get("confidence", 0.0)
        action = decision.get("action", "flag")
        severity = decision.get("severity", 0)
        needs_review = decision.get("needs_review")
        drivers = _drivers_str(decision.get("top_features"))
        proba = _proba_str(decision.get("class_probabilities"))

        action_phrase = {
            "allow": "permitted to proceed",
            "flag":  "flagged for analyst review",
            "block": "autonomously blocked",
        }.get(action, action)

        if pred == "normal":
            summary = (
                f"Connection classified as **benign traffic** with "
                f"{conf:.0%} confidence and {action_phrase}."
            )
        else:
            summary = (
                f"A **{desc}** ({pred.upper()}) was detected with {conf:.0%} "
                f"confidence. The connection was {action_phrase}."
            )

        next_steps = {
            "block": "Confirm the block held and review the source host for related activity.",
            "flag":  "Triage this alert: validate against logs and confirm or dismiss within SLA.",
            "allow": "No action required; retained for audit trail.",
        }.get(action, "Review per standard procedure.")
        if needs_review:
            next_steps += " Marked as requiring human verification."

        return (
            f"### Summary\n{summary}\n\n"
            f"### Severity & Action\nSeverity **{severity}/4** — action: **{action.upper()}**. "
            f"Class probabilities: {proba}.\n\n"
            f"### Key Indicators\nTop contributing features: {drivers}.\n\n"
            f"### Recommended Next Steps\n{next_steps}\n\n"
            f"_Generated by SAINT template engine (LLM offline)._"
        )
