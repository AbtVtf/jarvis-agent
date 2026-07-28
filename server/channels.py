"""Channels: let ANY local agent/process talk to the user through Jarvis.

An external session (e.g. an interactive Claude Code session in some repo)
POSTs a message to its named channel -> Jarvis speaks it with attribution.
If the sender expects a reply it long-polls /wait; when the user answers,
Jarvis routes it back and the sender's poll returns with the text.

This is deliberately plain HTTP on localhost so a single curl works from
anywhere; the `jarvis-chat` helper script wraps it.
"""

import asyncio
import time


class Channel:

  def __init__(self, name: str):
    self.name = name
    self.last_seen = time.time()
    self.last_message = ""
    self.awaiting_reply = False
    self.active_waiters = 0
    # At-least-once delivery: replies stay here until the client ACKs them,
    # so a poll whose client died can never swallow a reply.
    self.pending: list[dict] = []
    self.next_id = 1
    self.event = asyncio.Event()


class Channels:

  def __init__(self):
    self._channels: dict[str, Channel] = {}

  def _get(self, name: str) -> Channel:
    name = name.strip().lower()[:60] or "unnamed"
    if name not in self._channels:
      self._channels[name] = Channel(name)
    ch = self._channels[name]
    ch.last_seen = time.time()
    return ch

  def post(self, name: str, text: str, expect_reply: bool) -> Channel:
    ch = self._get(name)
    ch.last_message = text[:2000]
    ch.awaiting_reply = expect_reply
    return ch

  async def wait_for_reply(self, name: str, timeout_s: float) -> dict | None:
    """Returns the oldest un-ACKed reply WITHOUT removing it (client ACKs)."""
    ch = self._get(name)
    ch.awaiting_reply = True
    ch.active_waiters += 1
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    try:
      while True:
        if ch.pending:
          return ch.pending[0]
        remaining = deadline - loop.time()
        if remaining <= 0:
          return None
        ch.event.clear()
        try:
          await asyncio.wait_for(ch.event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
          return None
    finally:
      ch.active_waiters -= 1

  def ack(self, name: str, reply_id: int) -> bool:
    ch = self._get(name)
    before = len(ch.pending)
    ch.pending = [p for p in ch.pending if p["id"] != reply_id]
    if not ch.pending:
      ch.awaiting_reply = False
    return len(ch.pending) < before

  def reply(self, name: str, text: str) -> str:
    match = self.resolve(name)
    if not match:
      known = ", ".join(sorted(self._channels)) or "(none)"
      return f"No channel matching '{name}'. Known channels: {known}"
    match.pending.append({"id": match.next_id, "reply": text})
    match.next_id += 1
    match.event.set()
    if match.active_waiters > 0:
      return f"Delivered to the {match.name} session — it is listening."
    return (f"Queued for {match.name}. That session is not polling at this "
            "exact moment, but the message is held safely and will be "
            "delivered the next time it checks the channel.")

  def resolve(self, name: str) -> Channel | None:
    name = name.strip().lower()
    if name in self._channels:
      return self._channels[name]
    hits = [c for n, c in self._channels.items() if name in n]
    return hits[0] if len(hits) == 1 else None

  def overview(self) -> list[dict]:
    now = time.time()
    out = []
    for ch in sorted(self._channels.values(), key=lambda c: -c.last_seen):
      if now - ch.last_seen > 6 * 3600:
        continue
      out.append({
          "channel": ch.name,
          "awaiting_user_reply": ch.awaiting_reply,
          "listening_now": ch.active_waiters > 0,
          "undelivered_replies": len(ch.pending),
          "last_message": ch.last_message[:200],
          "minutes_since_active": int((now - ch.last_seen) / 60),
      })
    return out
