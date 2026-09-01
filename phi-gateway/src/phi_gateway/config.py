"""Policy loading. Every leak/utility knob comes from ``configs/policy.yaml`` so the
trade-off can be swept and measured rather than argued about."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .types import Category

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "policy.yaml"


@dataclass
class CategoryRule:
    action: str = "mask"          # mask | retain
    threshold: float = 0.30
    hipaa: int = 18

    @property
    def masks(self) -> bool:
        return self.action == "mask"


@dataclass
class DatePolicy:
    mode: str = "interval_preserving"   # safe_harbor | interval_preserving | shifted
    retain_year: bool = False
    shift_window_days: int = 180
    emit_interval_annotations: bool = True


@dataclass
class AgePolicy:
    retain_under_90: bool = True
    over_89_placeholder: str = "[AGE_OVER_89]"


@dataclass
class GeoPolicy:
    zip_strategy: str = "truncate3"
    restricted_zip3: tuple[str, ...] = ()


@dataclass
class NeuralPolicy:
    enabled: bool = False
    model_path: str = "artifacts/tagger"
    device: str = "cpu"
    max_length: int = 512
    stride: int = 128
    structured_segment_threshold_scale: float = 0.6


@dataclass
class SelfCheckPolicy:
    enabled: bool = True
    on_leak: str = "block"              # block | warn
    paranoid_threshold: float = 0.10


@dataclass
class ReviewQueuePolicy:
    enabled: bool = True
    low: float = 0.20
    high: float = 0.55


@dataclass
class VaultPolicy:
    ttl_seconds: int = 3600
    key_env: str = "PHI_VAULT_KEY"


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    prefer: tuple[str, ...] = ()

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    @property
    def available(self) -> bool:
        return bool(self.api_key)


@dataclass
class LLMPolicy:
    providers: tuple[ProviderConfig, ...] = ()
    timeout_seconds: int = 60
    max_retries: int = 2


@dataclass
class Policy:
    dates: DatePolicy = field(default_factory=DatePolicy)
    ages: AgePolicy = field(default_factory=AgePolicy)
    geo: GeoPolicy = field(default_factory=GeoPolicy)
    categories: dict[Category, CategoryRule] = field(default_factory=dict)
    neural: NeuralPolicy = field(default_factory=NeuralPolicy)
    selfcheck: SelfCheckPolicy = field(default_factory=SelfCheckPolicy)
    review_queue: ReviewQueuePolicy = field(default_factory=ReviewQueuePolicy)
    vault: VaultPolicy = field(default_factory=VaultPolicy)
    llm: LLMPolicy = field(default_factory=LLMPolicy)

    def rule(self, category: Category) -> CategoryRule:
        return self.categories.get(category, CategoryRule())

    def threshold(self, category: Category, *, in_structured_segment: bool = False) -> float:
        """Detection threshold for a category.

        PHI density inside headers, footers, tables and signature blocks is high, so we
        get *more* suspicious there rather than less -- the threshold is scaled down.
        """
        t = self.rule(category).threshold
        if in_structured_segment:
            t *= self.neural.structured_segment_threshold_scale
        return t

    def masks(self, category: Category) -> bool:
        return self.rule(category).masks


def load_policy(path: str | Path | None = None) -> Policy:
    p = Path(path) if path else DEFAULT_POLICY_PATH
    if not p.exists():
        return _with_default_categories(Policy())
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    cats: dict[Category, CategoryRule] = {}
    for name, cfg in (raw.get("categories") or {}).items():
        try:
            cat = Category(name)
        except ValueError:
            continue
        cats[cat] = CategoryRule(
            action=cfg.get("action", "mask"),
            threshold=float(cfg.get("threshold", 0.30)),
            hipaa=int(cfg.get("hipaa", 18)),
        )

    providers = tuple(
        ProviderConfig(
            name=pc["name"],
            base_url=pc["base_url"],
            api_key_env=pc.get("api_key_env", ""),
            prefer=tuple(pc.get("prefer", ())),
        )
        for pc in (raw.get("llm", {}).get("providers") or [])
    )

    policy = Policy(
        dates=DatePolicy(**(raw.get("dates") or {})),
        ages=AgePolicy(**(raw.get("ages") or {})),
        geo=GeoPolicy(
            zip_strategy=(raw.get("geo") or {}).get("zip_strategy", "truncate3"),
            restricted_zip3=tuple((raw.get("geo") or {}).get("restricted_zip3", ())),
        ),
        categories=cats,
        neural=NeuralPolicy(**(raw.get("neural") or {})),
        selfcheck=SelfCheckPolicy(**(raw.get("selfcheck") or {})),
        review_queue=ReviewQueuePolicy(**(raw.get("review_queue") or {})),
        vault=VaultPolicy(**(raw.get("vault") or {})),
        llm=LLMPolicy(
            providers=providers,
            timeout_seconds=int((raw.get("llm") or {}).get("timeout_seconds", 60)),
            max_retries=int((raw.get("llm") or {}).get("max_retries", 2)),
        ),
    )
    return _with_default_categories(policy)


def _with_default_categories(policy: Policy) -> Policy:
    """Any category missing from the YAML defaults to mask-at-0.30 rather than to
    retain. Fail closed: an unconfigured identifier must not silently pass through."""
    for cat in Category:
        policy.categories.setdefault(cat, CategoryRule(action="mask", threshold=0.30))
    return policy
