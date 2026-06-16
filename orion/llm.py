"""Thin Anthropic wrapper used by agents to *interpret* (never compute) numbers.

Degrades gracefully: if no ANTHROPIC_API_KEY or the SDK isn't installed, callers
fall back to deterministic rule-based narratives, so the whole pipeline still runs
offline on sample data.
"""
from __future__ import annotations

from .config import Settings


class LLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        if settings.has_anthropic:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def interpret(
        self, system: str, prompt: str, model: str | None = None, max_tokens: int = 700
    ) -> str | None:
        """Return the model's text, or None if the LLM is unavailable/errors."""
        if not self._client:
            return None
        try:
            resp = self._client.messages.create(
                model=model or self.settings.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()
        except Exception:
            return None
