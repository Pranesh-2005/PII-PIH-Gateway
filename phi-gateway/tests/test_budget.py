"""P2.3 hard requirement: the tagger must be under 1B parameters, with the exact count
reported. This test *is* the report -- run ``pytest -s tests/test_budget.py`` and the count
prints.

Skips cleanly until Phase 3 produces a checkpoint, so the suite stays green on a machine
with no torch installed. A skip is honest; a hardcoded "149M" in the README is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phi_gateway.config import load_policy

PARAM_BUDGET = 1_000_000_000


def _model_dir() -> Path | None:
    policy = load_policy()
    root = Path(__file__).resolve().parents[1]
    candidate = (root / policy.neural.model_path).resolve()
    return candidate if candidate.exists() else None


def test_param_count_under_budget():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    model_dir = _model_dir()
    if model_dir is None:
        pytest.skip("no trained tagger yet (Phase 3); nothing to weigh")

    model = transformers.AutoModelForTokenClassification.from_pretrained(str(model_dir))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\ntagger: {model_dir}")
    print(f"  parameters (total):     {total:,}")
    print(f"  parameters (trainable): {trainable:,}")
    print(f"  budget:                 {PARAM_BUDGET:,}")
    assert total < PARAM_BUDGET, f"{total:,} parameters exceeds the 1B limit"
