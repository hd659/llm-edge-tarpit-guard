"""3-strike sliding-window state machine + progressive latency SSE tarpit.

Flow per request: ``Verdict`` (from ModerationFilter) -> record strike weight in
a per-client sliding window -> derive state -> ``RouteDecision``.

States::

    CLEAN      strikes == 0                 -> pass, no delay
    PROBATION  0 < strikes < strike_limit   -> pass, no delay (warning header)
    TARPIT     strikes >= strike_limit      -> pass, delay = base * mult**level

``block`` verdicts are always rejected (and add a full strike); ``flag`` adds a
fractional strike. ``level`` grows by one for every strike received while in
TARPIT, so persistent violators see exponentially slower streams, capped at
``max_delay``. Strikes expire after ``window_s``; the client decays back to
CLEAN without any explicit reset. Delay is paid as SSE keep-alive comments so
the connection looks alive but throughput collapses (cheap for us, costly for
automated abusers).
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Awaitable, Callable, Iterable, Iterator

from edge_guard.moderation_filter import Verdict

__all__ = ["State", "TarpitPolicy", "RouteDecision", "TarpitRouter", "tarpit_sse", "atarpit_sse"]


class State(str, Enum):
    CLEAN = "clean"
    PROBATION = "probation"
    TARPIT = "tarpit"


@dataclass(frozen=True)
class TarpitPolicy:
    strike_limit: float = 3.0
    window_s: float = 600.0
    flag_weight: float = 0.5
    block_weight: float = 1.0
    base_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 30.0
    max_clients: int = 10_000

    def __post_init__(self) -> None:
        if self.strike_limit <= 0 or self.window_s <= 0 or self.max_clients <= 0:
            raise ValueError("strike_limit, window_s and max_clients must be > 0")
        if self.multiplier < 1 or not 0 <= self.base_delay <= self.max_delay:
            raise ValueError("require multiplier >= 1 and 0 <= base_delay <= max_delay")


@dataclass
class RouteDecision:
    action: str  # "pass" | "tarpit" | "reject"
    state: State
    strikes: float
    delay_s: float = 0.0
    reason: str = ""

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Guard-State": self.state.value,
                "X-Guard-Strikes": f"{self.strikes:g}"}


@dataclass
class _Client:
    events: deque = field(default_factory=deque)  # (timestamp, weight)
    level: int = 0


class TarpitRouter:
    def __init__(self, policy: TarpitPolicy | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.policy = policy or TarpitPolicy()
        self._clock = clock
        self._clients: OrderedDict[str, _Client] = OrderedDict()

    def _client(self, key: str) -> _Client:
        c = self._clients.get(key)
        if c is None:
            c = self._clients[key] = _Client()
            while len(self._clients) > self.policy.max_clients:
                self._clients.popitem(last=False)  # LRU eviction
        else:
            self._clients.move_to_end(key)
        return c

    def _expire(self, c: _Client, now: float) -> float:
        horizon = now - self.policy.window_s
        while c.events and c.events[0][0] <= horizon:
            c.events.popleft()
        total = round(sum(w for _, w in c.events), 6)
        if total < self.policy.strike_limit:
            c.level = 0  # decayed out of the tarpit
        return total

    def _state(self, strikes: float) -> State:
        if strikes >= self.policy.strike_limit:
            return State.TARPIT
        return State.PROBATION if strikes > 0 else State.CLEAN

    def delay_for(self, level: int) -> float:
        p = self.policy
        return min(p.max_delay, p.base_delay * p.multiplier ** max(level, 0))

    def strikes(self, key: str) -> float:
        c = self._clients.get(key)
        return self._expire(c, self._clock()) if c else 0.0

    def state(self, key: str) -> State:
        return self._state(self.strikes(key))

    def reset(self, key: str) -> None:
        self._clients.pop(key, None)

    def route(self, key: str, verdict: Verdict) -> RouteDecision:
        p, now = self.policy, self._clock()
        c = self._client(key)
        was_tarpit = self._expire(c, now) >= p.strike_limit
        weight = {"block": p.block_weight, "flag": p.flag_weight}.get(verdict.action, 0.0)
        if weight > 0:
            c.events.append((now, weight))
            if was_tarpit:
                c.level += 1
        strikes = self._expire(c, now)
        state = self._state(strikes)
        if verdict.action == "block":
            return RouteDecision("reject", state, strikes, 0.0, "blocked:" + ",".join(verdict.hits))
        if state is State.TARPIT:
            return RouteDecision("tarpit", state, strikes, self.delay_for(c.level),
                                 f"tarpit:level={c.level}")
        return RouteDecision("pass", state, strikes, 0.0, verdict.action)


def _sse(data: str) -> str:
    return "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"


def _ticks(delay: float, tick: float) -> list[float]:
    if delay <= 0:
        return []
    n = max(1, int(-(-delay // tick)))
    return [delay / n] * n


def tarpit_sse(chunks: Iterable[str], delay_s: float, *, tick_s: float = 1.0,
               sleep: Callable[[float], None] = time.sleep) -> Iterator[str]:
    """Wrap upstream chunks as SSE; pay ``delay_s`` before *each* chunk via keep-alives."""
    for chunk in chunks:
        for t in _ticks(delay_s, tick_s):
            sleep(t)
            yield ": tarpit\n\n"
        yield _sse(chunk)
    yield _sse("[DONE]")


async def atarpit_sse(chunks: Iterable[str], delay_s: float, *, tick_s: float = 1.0,
                      sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> AsyncIterator[str]:
    """Async twin of :func:`tarpit_sse` for ASGI StreamingResponse."""
    for chunk in chunks:
        for t in _ticks(delay_s, tick_s):
            await sleep(t)
            yield ": tarpit\n\n"
        yield _sse(chunk)
    yield _sse("[DONE]")
