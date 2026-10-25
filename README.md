# llm-edge-tarpit-guard

Zero-latency, code-aware LLM moderation filter with a 3-strike SSE tarpit router.

Two pure-Python stages sit in front of an LLM gateway:

1. `edge_guard.moderation_filter`: removes Markdown code (fenced, inline, indented) and tool payloads (`tool_result` / `tool_use` blocks, `role: tool` messages, XML tool tags) **before** scoring. A weighted noisy-OR rule set then scores what is left. Code like `kill -9`, `DROP TABLE` or `exploit()` inside a fence can never cause a false positive.
2. `edge_guard.tarpit_router`: a per-client sliding-window strike state machine. Repeat violators are not hard-banned. Their streams are slowed down progressively, which costs automated abusers a lot and costs the gateway very little.

No network calls and no ML model on the hot path, so the guard adds only microseconds of regex work per request.

## Layout

```
edge_guard/
  moderation_filter.py   strip_code_context, extract_text, ModerationFilter, Verdict
  tarpit_router.py       TarpitPolicy, TarpitRouter, RouteDecision, tarpit_sse, atarpit_sse
tests/
  test_moderation_filter.py
  test_tarpit_router.py
```

## Tarpit state machine

| State     | Condition                        | Effect                                  |
|-----------|----------------------------------|-----------------------------------------|
| CLEAN     | strikes == 0                     | pass, no delay                          |
| PROBATION | 0 < strikes < strike_limit       | pass, `X-Guard-State: probation` header |
| TARPIT    | strikes >= strike_limit          | pass with `delay = base * mult ** level`, capped at `max_delay` |

- `block` verdict: always rejected, adds `block_weight` (default 1.0) strikes.
- `flag` verdict: still served, adds `flag_weight` (default 0.5) strikes.
- Each new strike taken while already in TARPIT raises `level` by 1, so the delay doubles (1s, 2s, 4s ... up to 30s by default).
- Strikes expire after `window_s` (default 600s). Once the total drops below the limit, `level` resets and the client is back to normal without any manual unban.
- The client table is an LRU capped at `max_clients`, so memory stays bounded under key-spraying.

Cost model: an abuser running `N` requests in the window pays roughly `sum(base * mult ** k)` seconds of wall-clock. The gateway only pays for one idle connection that sends a keep-alive comment every `tick_s`.

## Usage

```python
from edge_guard.moderation_filter import ModerationFilter
from edge_guard.tarpit_router import TarpitRouter, TarpitPolicy, atarpit_sse

mf = ModerationFilter()
router = TarpitRouter(TarpitPolicy(strike_limit=3, window_s=600))

async def handle(client_id: str, payload: dict, upstream_chunks):
    decision = router.route(client_id, mf.evaluate(payload))
    if decision.action == "reject":
        return 400, decision.headers, decision.reason
    # decision.delay_s == 0 for pass; > 0 for tarpit
    stream = atarpit_sse(upstream_chunks, decision.delay_s, tick_s=1.0)
    return 200, {**decision.headers, "Content-Type": "text/event-stream"}, stream
```

The sync version `tarpit_sse` has the same signature for WSGI / generator servers. Both accept an injectable `sleep`, and the router accepts an injectable `clock`, so tests run deterministically in milliseconds.

The delay is paid as SSE comment lines (`: tarpit`), which clients ignore. The connection stays open and well-formed, but throughput drops, and the client gets no clear signal that it has been rate-limited.

## Testing

```bash
python -m pytest -q
```

The suite covers stripping edge cases, scorer saturation, strike accumulation, fractional flags, exponential delay with its cap, sliding-window expiry and decay, LRU eviction, policy validation, SSE framing (sync and async), and an end-to-end check that code-heavy prompts stay CLEAN while jailbreak prompts reach TARPIT in three requests.

## Roadmap

- `let-m1`: code-aware stripper and weighted risk scorer (done)
- `let-m2`: 3-strike sliding-window router and progressive SSE tarpit (done)
- next: shared-state backend (Redis) for multi-instance gateways, plus ASGI middleware
