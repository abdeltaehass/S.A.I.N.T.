"""
Feature attribution for SAINT decisions.

Uses Integrated Gradients (Sundararajan et al., 2017) against a zero baseline
to attribute the predicted-class logit back onto the 122-dim feature vector.
The post-encoding column space is then folded back into the 41 raw NSL-KDD
features so analysts see "src_bytes ↑, flag=REJ, service=private" instead of
"feature_47 = 0.83".
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Friendly-name resolver
# ---------------------------------------------------------------------------

_ENCODED_RE = re.compile(r"^(protocol_type|service|flag)_(\d+)$")


def build_friendly_names(
    feature_cols: list[str],
    encoders: dict | None,
) -> list[str]:
    """
    Map post-encoding column names → analyst-friendly labels.

      "service_42"  →  "service=http"   (if encoder maps index 42 → "http")
      "flag_3"      →  "flag=REJ"
      "src_bytes"   →  "src_bytes"      (numeric, unchanged)
    """
    friendly: list[str] = []
    for col in feature_cols:
        m = _ENCODED_RE.match(col)
        if m and encoders and m.group(1) in encoders:
            cat, idx = m.group(1), int(m.group(2))
            classes = encoders[cat].classes_
            label = classes[idx] if idx < len(classes) else f"#{idx}"
            friendly.append(f"{cat}={label}")
        else:
            friendly.append(col)
    return friendly


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class FeatureExplainer:
    """
    Computes top-k feature attributions for a single connection using
    Integrated Gradients. Designed to be called only on flag/block decisions
    so the fast path (allow) skips the extra forward passes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_cols: list[str],
        encoders: dict | None = None,
        device: str | None = None,
        steps: int = 16,
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps
        self.feature_cols = feature_cols
        self.friendly_names = build_friendly_names(feature_cols, encoders)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def top_features(
        self,
        x: np.ndarray,
        target_class: int,
        k: int = 5,
    ) -> list[dict]:
        """
        Return the top-k features that drove the score for `target_class`,
        ranked by absolute attribution. Each entry:

            {"name": "src_bytes", "value": 1.23, "contribution": 0.41}

        `value` is the (scaled) feature value as fed to the model, so
        analysts can see whether the feature was unusually high or low
        relative to the training distribution mean of zero.
        """
        attribution = self._integrated_gradients(x, target_class)
        x_flat = np.asarray(x, dtype=np.float32).reshape(-1)

        order = np.argsort(np.abs(attribution))[::-1]
        out: list[dict] = []
        for i in order:
            # Skip features that contribute essentially nothing
            if abs(attribution[i]) < 1e-6:
                continue
            out.append({
                "name":         self.friendly_names[i],
                "raw_name":     self.feature_cols[i],
                "value":        float(x_flat[i]),
                "contribution": float(attribution[i]),
            })
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------------------

    def _integrated_gradients(
        self,
        x: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """
        IG attribution = (x - baseline) ⊙ ∫₀¹ ∂F(baseline + α·(x-baseline))/∂x dα

        Approximated with `self.steps` Riemann-midpoint samples. Baseline is
        the all-zeros vector — natural here because numeric features are
        StandardScaled (so zero ≈ training mean) and one-hot indicators
        are zero when the category is absent.
        """
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).reshape(1, -1)
        baseline = torch.zeros_like(x_t)

        # Riemann-midpoint alphas: (1/2N, 3/2N, ..., (2N-1)/2N)
        alphas = (torch.arange(self.steps, device=self.device, dtype=torch.float32) + 0.5) / self.steps
        # (steps, 1, D)
        interp = baseline + alphas.view(-1, 1, 1) * (x_t - baseline).unsqueeze(0)
        interp = interp.squeeze(1).clone().detach().requires_grad_(True)

        was_training = self.model.training
        self.model.eval()
        logits = self.model(interp)              # (steps, num_classes)
        target_score = logits[:, target_class].sum()
        grads = torch.autograd.grad(target_score, interp)[0]   # (steps, D)
        if was_training:
            self.model.train()

        avg_grad = grads.mean(dim=0)             # (D,)
        attribution = (x_t.squeeze(0) - baseline.squeeze(0)) * avg_grad
        return attribution.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Formatting helper
    # ------------------------------------------------------------------

    @staticmethod
    def format_summary(top: Iterable[dict], max_items: int = 3) -> str:
        """Compact one-liner like 'src_bytes↑, flag=REJ↑, count↓'."""
        parts = []
        for f in list(top)[:max_items]:
            arrow = "↑" if f["contribution"] > 0 else "↓"
            parts.append(f"{f['name']}{arrow}")
        return ", ".join(parts)
