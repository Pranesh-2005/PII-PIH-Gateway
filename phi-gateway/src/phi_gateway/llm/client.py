"""Foundation LLM client.

Groq and NVIDIA NIM are both OpenAI-compatible, so one client covers both and failover is
just "try the next base_url".

Model IDs are **discovered at runtime** via ``GET /models`` and matched against a preference
list, never hardcoded. Both providers rotate their catalogues on a timescale shorter than a
hiring process, and a 404 on a dead model ID during the live demo is an entirely avoidable
crash.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import LLMPolicy, ProviderConfig


class LLMUnavailable(RuntimeError):
    """Every configured provider failed. The caller should degrade, not crash."""


#: Model families that answer chat completions badly or not at all.
_NOT_CHAT = re.compile(
    r"(?i)(embed|embedding|rerank|whisper|tts|audio|guard|moderation|vision-ocr|"
    r"retrieval|nemoretriever|clip|sana|flux|riva|parakeet|canary)"
)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


@dataclass
class LLMClient:
    policy: LLMPolicy
    _resolved: dict[str, str] = field(default_factory=dict, repr=False)
    _errors: list[str] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ discovery
    def _sdk(self, provider: ProviderConfig):
        from openai import OpenAI

        return OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key or "missing",
            timeout=self.policy.timeout_seconds,
            max_retries=self.policy.max_retries,
        )

    def available_providers(self) -> list[ProviderConfig]:
        return [p for p in self.policy.providers if p.available]

    def list_models(self, provider: ProviderConfig) -> list[str]:
        try:
            return [m.id for m in self._sdk(provider).models.list().data]
        except Exception as exc:                      # network, auth, endpoint shape
            self._errors.append(f"{provider.name}: models.list failed: {exc}")
            return []

    def resolve_model(self, provider: ProviderConfig) -> str | None:
        """First preferred model the endpoint actually serves, else the first plausible
        chat model it reports, else the raw preference (let the API be the authority)."""
        if provider.name in self._resolved:
            return self._resolved[provider.name]

        served = self.list_models(provider)
        chosen: str | None = None
        for want in provider.prefer:
            for got in served:
                if got == want or want.lower() in got.lower():
                    chosen = got
                    break
            if chosen:
                break
        if chosen is None:
            chosen = next((m for m in served if not _NOT_CHAT.search(m)), None)
        if chosen is None and provider.prefer:
            chosen = provider.prefer[0]

        if chosen:
            self._resolved[provider.name] = chosen
        return chosen

    # ------------------------------------------------------------------ completion
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LLMResult:
        self._errors = []
        providers = self.available_providers()
        if not providers:
            envs = ", ".join(p.api_key_env for p in self.policy.providers) or "none configured"
            raise LLMUnavailable(f"no provider has an API key set (looked for: {envs})")

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]

        for provider in providers:
            model = self.resolve_model(provider)
            if not model:
                self._errors.append(f"{provider.name}: no usable model found")
                continue
            try:
                resp = self._sdk(provider).chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return LLMResult(
                    text=resp.choices[0].message.content or "",
                    provider=provider.name,
                    model=model,
                )
            except Exception as exc:
                # Rate limit, dead model, transport error -- all mean "try the next one".
                self._errors.append(f"{provider.name}/{model}: {type(exc).__name__}: {exc}")
                self._resolved.pop(provider.name, None)   # re-discover next attempt

        raise LLMUnavailable("all providers failed:\n  " + "\n  ".join(self._errors))
