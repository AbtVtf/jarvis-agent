"""Claude Code sessions: watch turns, read transcripts, reply via CloudCLI.

The integration is push-based and needs zero cooperation from Claude:
a global Stop hook (~/.claude/settings.json) POSTs to /claude-event whenever
ANY Claude Code session finishes a turn. Jarvis reads the last assistant
message straight from the transcript JSONL on disk and composes his own
spoken update.

Replies go through CloudCLI's chat gateway (chat.send over its websocket),
so the conversation continues visibly in the CloudCLI UI and, to Claude,
the user's words are just the next prompt. CloudCLI indexes CLI-started
sessions under their Claude-native session id, so the id from the Stop hook
is directly addressable.
"""

import asyncio
import json
import os
import time

import httpx
import websockets

from server import config

CLAUDE_PROJECTS = os.path.expanduser("~/.claude/projects")


def _last_assistant_text(transcript_path: str) -> str:
  """Last assistant message with visible text (tool-only turns skipped)."""
  best = ""
  try:
    with open(transcript_path, encoding="utf-8") as f:
      for line in f:
        try:
          entry = json.loads(line)
        except json.JSONDecodeError:
          continue
        if entry.get("type") != "assistant":
          continue
        content = (entry.get("message") or {}).get("content") or []
        if isinstance(content, str):
          text = content
        else:
          text = "\n".join(b.get("text", "") for b in content
                           if isinstance(b, dict) and b.get("type") == "text")
        if text.strip():
          best = text.strip()
  except OSError:
    pass
  return best


def _project_of(transcript_path: str) -> str:
  """Project folder name via the cwd recorded inside the transcript."""
  try:
    with open(transcript_path, encoding="utf-8") as f:
      for i, line in enumerate(f):
        if i > 20:
          break
        try:
          cwd = json.loads(line).get("cwd")
        except json.JSONDecodeError:
          continue
        if cwd:
          return os.path.basename(cwd.rstrip("/"))
  except OSError:
    pass
  return os.path.basename(os.path.dirname(transcript_path)).split("-")[-1]


def _tail_messages(transcript_path: str, count: int) -> list[dict]:
  """Last `count` user/assistant text messages, oldest first."""
  out = []
  try:
    with open(transcript_path, encoding="utf-8") as f:
      for line in f:
        try:
          entry = json.loads(line)
        except json.JSONDecodeError:
          continue
        role = entry.get("type")
        if role not in ("user", "assistant"):
          continue
        content = (entry.get("message") or {}).get("content") or []
        if isinstance(content, str):
          text = content
        else:
          text = "\n".join(b.get("text", "") for b in content
                           if isinstance(b, dict) and b.get("type") == "text")
        text = text.strip()
        if text:
          out.append({"role": role, "text": text[:2000]})
  except OSError:
    pass
  return out[-count:]


class ClaudeSessions:
  """Registry of announced turns + CloudCLI client for replies."""

  def __init__(self):
    self.client = httpx.AsyncClient(timeout=15)
    self._token = None
    # Turns announced this server run, newest last; drives the brain's
    # context so "tell them yes" right after an announcement has a target.
    self.recent: list[dict] = []

  # -- turn intake (Stop hook) --

  async def record_turn(self, session_id: str, cwd: str,
                        transcript_path: str) -> dict:
    """Wait for the turn's FINAL message to land in the transcript.

    The Stop hook fires before Claude Code flushes the last assistant
    message to disk, so a single immediate read announces the PREVIOUS
    turn. Poll until the transcript's last text entry is an assistant
    message we haven't announced yet (a trailing user entry means the
    reply is still being written); give up quietly after ~10 s.
    """
    project = os.path.basename(cwd.rstrip("/")) or cwd
    prev = next((r["last_text"] for r in self.recent
                 if r["session_id"] == session_id), None)
    text, last_user, tail, last_size = "", "", [], -1
    for _ in range(20):
      try:
        size = os.path.getsize(transcript_path)
      except OSError:
        size = -2
      stable = size == last_size
      last_size = size
      tail = _tail_messages(transcript_path, 2)
      fresh = (tail and tail[-1]["role"] == "assistant"
               and tail[-1]["text"] != prev)
      if fresh and stable:
        break
      await asyncio.sleep(0.5)
    if tail and tail[-1]["role"] == "assistant" and tail[-1]["text"] != prev:
      text = tail[-1]["text"]
      if len(tail) > 1 and tail[0]["role"] == "user":
        last_user = tail[0]["text"]
    info = {"session_id": session_id, "project": project,
            "transcript_path": transcript_path, "cwd": cwd,
            "last_text": text, "last_user": last_user, "at": time.time()}
    self.recent = [r for r in self.recent
                   if r["session_id"] != session_id][-11:] + [info]
    return info

  def overview(self) -> list[dict]:
    """Recently active sessions across all projects (transcript scan)."""
    found = []
    cutoff = time.time() - 48 * 3600
    try:
      project_dirs = os.listdir(CLAUDE_PROJECTS)
    except OSError:
      return []
    for d in project_dirs:
      full = os.path.join(CLAUDE_PROJECTS, d)
      try:
        files = [os.path.join(full, f) for f in os.listdir(full)
                 if f.endswith(".jsonl")]
      except OSError:
        continue
      for path in files:
        try:
          mtime = os.path.getmtime(path)
        except OSError:
          continue
        if mtime < cutoff:
          continue
        found.append({"session_id": os.path.splitext(
            os.path.basename(path))[0],
            "minutes_ago": int((time.time() - mtime) / 60),
            "transcript_path": path})
    found.sort(key=lambda s: s["minutes_ago"])
    for s in found[:10]:
      s["project"] = _project_of(s["transcript_path"])
      s["last_message"] = _last_assistant_text(s["transcript_path"])[:300]
    return [{k: v for k, v in s.items() if k != "transcript_path"}
            for s in found[:10]]

  def read_session(self, session_id: str, count: int = 6) -> list[dict]:
    info = self._resolve(session_id)
    if not info:
      return []
    return _tail_messages(info["transcript_path"], count)

  def _resolve(self, ref: str) -> dict | None:
    """Find a session by full id, id prefix, or project name."""
    ref = (ref or "").strip().lower()
    for r in reversed(self.recent):
      if r["session_id"].lower().startswith(ref) or ref in r["project"].lower():
        return r
    # Fall back to a disk scan for sessions never announced this run:
    # match session-id prefix or project name, newest transcript wins.
    candidates = []
    for d in os.listdir(CLAUDE_PROJECTS):
      full = os.path.join(CLAUDE_PROJECTS, d)
      if not os.path.isdir(full):
        continue
      dir_matches = ref and ref.replace(" ", "-") in d.lower()
      for f in os.listdir(full):
        if not f.endswith(".jsonl"):
          continue
        if dir_matches or f.lower().startswith(ref):
          path = os.path.join(full, f)
          try:
            candidates.append((os.path.getmtime(path), path, f[:-6]))
          except OSError:
            continue
    if not candidates:
      return None
    _, path, sid = max(candidates)
    return {"session_id": sid, "project": _project_of(path),
            "transcript_path": path}

  # -- CloudCLI reply path --

  async def _login(self) -> str:
    resp = await self.client.post(
        f"{config.CLOUDCLI_URL}/api/auth/login",
        json={"username": config.CLOUDCLI_USER,
              "password": config.CLOUDCLI_PASS})
    resp.raise_for_status()
    self._token = resp.json()["token"]
    return self._token

  async def send(self, session_ref: str, text: str) -> str:
    info = self._resolve(session_ref)
    if not info:
      return (f"no session matching '{session_ref}' found; "
              "check sessions_overview")
    if not self._token:
      await self._login()
    ws_url = (config.CLOUDCLI_URL.replace("http", "ws", 1)
              + f"/ws?token={self._token}")
    try:
      reply = await self._ws_send(ws_url, info["session_id"], text)
    except websockets.InvalidStatus:
      await self._login()  # token expired -> one retry with a fresh one
      ws_url = (config.CLOUDCLI_URL.replace("http", "ws", 1)
                + f"/ws?token={self._token}")
      reply = await self._ws_send(ws_url, info["session_id"], text)
    return reply

  async def _ws_send(self, ws_url: str, session_id: str, text: str) -> str:
    """Returns an honest delivery verdict.

    CloudCLI rejections arrive as {kind: 'protocol_error', code, error} —
    NO 'type' field — and run activity only reaches subscribers, so we
    subscribe before sending and wait for evidence either way.
    """
    async with websockets.connect(ws_url) as ws:
      await ws.send(json.dumps({
          "type": "chat.send", "sessionId": session_id, "content": text,
          # Voice replies must never stall on an approval prompt the user
          # can't see; CloudCLI's claude runtime maps this to
          # --dangerously-skip-permissions.
          "options": {"permissionMode": "bypassPermissions"}}))
      # Subscribe AFTER the send: the run is registered by then, so the
      # ack's isProcessing:true doubles as a delivery receipt. The field
      # is a `sessions` array — a bare sessionId is silently ignored.
      await ws.send(json.dumps({"type": "chat.subscribe",
                                "sessions": [{"sessionId": session_id}]}))
      # Verdict ladder: protocol_error => failed; run events => delivered;
      # chat_subscribed alone => accepted (a big session can take longer
      # than our window to emit its first event); silence => uncertain.
      accepted, saw_activity = False, False
      try:
        async with asyncio.timeout(6):
          while True:
            msg = json.loads(await ws.recv())
            kind = msg.get("kind") or msg.get("type") or ""
            if kind == "protocol_error":
              verdict = (f"NOT delivered — CloudCLI rejected it: "
                         f"{msg.get('error')} (code {msg.get('code')})")
              print(f"[sessions] send {session_id[:8]}: {verdict}", flush=True)
              return verdict
            if kind == "chat_subscribed":
              accepted = True
              if msg.get("isProcessing"):
                saw_activity = True
                break
            elif kind:
              saw_activity = True
              break
      except TimeoutError:
        pass
    if saw_activity:
      verdict = "delivered — the session is working on it now"
    elif accepted:
      verdict = ("delivered — CloudCLI accepted it and the session is "
                 "starting up")
    else:
      verdict = ("sent, but CloudCLI gave no confirmation — delivery is "
                 "UNCERTAIN; check the session before telling the user it "
                 "was delivered")
    print(f"[sessions] send {session_id[:8]}: {verdict}", flush=True)
    return verdict
