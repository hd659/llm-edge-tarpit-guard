"""Code-aware edge moderation filter.

Pipeline: raw prompt / message list -> strip code & tool payloads -> weighted
risk scoring -> verdict (allow / flag / block). Stripping happens *before*
scoring so source code, stack traces and tool outputs never trigger false
positives (e.g. ``kill -9``, ``exploit()`` or ``DROP TABLE`` inside a fence).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["RiskRule", "Verdict", "strip_code_context", "extract_text", "ModerationFilter"]

_FENCE_RE = re.compile(r"(?P<f>`{3,}|~{3,})[^\n]*\n.*?(?:\n(?P=f)[ \t]*(?=\n|$)|\Z)", re.S)
_INLINE_RE = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)
_TOOL_TAG_RE = re.compile(
    r"<(?P<t>tool_result|tool_use|function_results|function_calls|tool_output)\b[^>]*>.*?(?:</(?P=t)>|\Z)",
    re.S | re.I,
)
_INDENT_RE = re.compile(r"(?:^|\n)(?:(?: {4}|\t)[^\n]*(?:\n|$))+")
_TOOL_TYPES = frozenset({"tool_result", "tool_use", "function_call", "function_call_output"})
_PLACEHOLDER = " "


def strip_code_context(text: str, *, strip_indented: bool = True) -> str:
    """Remove fenced/inline code, tool XML payloads and indented code blocks."""
    if not text:
        return ""
    out = _TOOL_TAG_RE.sub(_PLACEHOLDER, text)
    out = _FENCE_RE.sub(_PLACEHOLDER, out)
    if strip_indented:
        out = _INDENT_RE.sub("\n", out)
    out = _INLINE_RE.sub(_PLACEHOLDER, out)
    return re.sub(r"[ \t]+", " ", out).strip()


def extract_text(payload: Any) -> str:
    """Flatten OpenAI/Anthropic-style messages into moderatable user text.

    Drops ``tool``/``function`` roles and ``tool_result``/``tool_use`` content
    blocks entirely; keeps only ``text`` blocks and plain strings.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        if payload.get("role") in {"tool", "function"} or payload.get("type") in _TOOL_TYPES:
            return ""
        if "messages" in payload:
            return extract_text(payload["messages"])
        if "content" in payload:
            return extract_text(payload["content"])
        return payload.get("text", "") if isinstance(payload.get("text"), str) else ""
    if isinstance(payload, Iterable):
        return "\n".join(t for t in (extract_text(p) for p in payload) if t)
    return ""


@dataclass(frozen=True)
class RiskRule:
    name: str
    category: str
    pattern: re.Pattern[str]
    weight: float

    @classmethod
    def build(cls, name: str, category: str, regex: str, weight: float) -> "RiskRule":
        return cls(name, category, re.compile(regex, re.I), weight)


DEFAULT_RULES: tuple[RiskRule, ...] = (
    RiskRule.build("ignore_prior", "jailbreak",
                   r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all)\b.{0,20}\b(instructions?|rules|prompts?)\b", 0.45),
    RiskRule.build("persona_dan", "jailbreak", r"\b(DAN|do anything now|developer mode|jailbr(eak|oken))\b", 0.40),
    RiskRule.build("no_limits", "jailbreak",
                   r"\b(no|without)\s+(restrictions|filters|limits|guidelines|censorship)\b", 0.25),
    RiskRule.build("system_leak", "jailbreak",
                   r"\b(reveal|print|show|repeat)\b.{0,25}\b(system prompt|hidden instructions|initial prompt)\b", 0.35),
    RiskRule.build("roleplay_evil", "jailbreak", r"\b(pretend|act as|roleplay)\b.{0,30}\b(unfiltered|evil|uncensored)\b", 0.30),
    RiskRule.build("weapon_synth", "violence",
                   r"\b(build|make|synthesi[sz]e|assemble)\b.{0,30}\b(bomb|explosive|nerve agent|bioweapon)\b", 0.70),
    RiskRule.build("self_harm", "self-harm", r"\b(kill myself|end my life|suicide method)\b", 0.60),
    RiskRule.build("malware_req", "illicit",
                   r"\b(write|create|generate)\b.{0,25}\b(ransomware|keylogger|credential stealer)\b", 0.50),
)


@dataclass
class Verdict:
    action: str
    score: float
    categories: dict[str, float] = field(default_factory=dict)
    hits: list[str] = field(default_factory=list)
    cleaned: str = ""

    @property
    def flagged(self) -> bool:
        return self.action != "allow"


class ModerationFilter:
    """Weighted, saturating risk scorer over code-stripped text.

    Per-category score uses noisy-OR: ``1 - prod(1 - w_i)`` so repeated weak
    signals accumulate but never exceed 1.0. Global score = max category.
    """

    def __init__(self, rules: Sequence[RiskRule] = DEFAULT_RULES, *,
                 flag_threshold: float = 0.35, block_threshold: float = 0.65,
                 category_bias: Mapping[str, float] | None = None) -> None:
        if not 0 < flag_threshold <= block_threshold <= 1:
            raise ValueError("require 0 < flag_threshold <= block_threshold <= 1")
        self.rules = tuple(rules)
        self.flag_threshold = flag_threshold
        self.block_threshold = block_threshold
        self.category_bias = dict(category_bias or {})

    def score_text(self, text: str) -> Verdict:
        cleaned = strip_code_context(text)
        residual: dict[str, float] = {}
        hits: list[str] = []
        for rule in self.rules:
            n = len(rule.pattern.findall(cleaned))
            if not n:
                continue
            hits.append(rule.name)
            w = min(1.0, rule.weight * self.category_bias.get(rule.category, 1.0))
            residual[rule.category] = residual.get(rule.category, 1.0) * (1.0 - w) ** min(n, 3)
        cats = {c: round(1.0 - r, 4) for c, r in residual.items()}
        score = max(cats.values(), default=0.0)
        if score >= self.block_threshold:
            action = "block"
        elif score >= self.flag_threshold:
            action = "flag"
        else:
            action = "allow"
        return Verdict(action, score, cats, hits, cleaned)

    def evaluate(self, payload: Any) -> Verdict:
        """Score a raw string, a message dict, or a list of messages."""
        return self.score_text(extract_text(payload))
