"""Jarvis's brain: LLM voice companion over the user's Claude Code sessions.

Backends:
  openrouter (default when a key exists) — OpenAI-compatible streaming with
    tool calls; model set by JARVIS_OPENROUTER_MODEL (deepseek-v4-flash).
  ollama — local fallback (gemma4); also used automatically per-turn when
    OpenRouter is unreachable, so Jarvis survives offline.

Jarvis never writes code and never drives Claude directly. Turn completions
arrive by themselves (Stop hook -> /claude-event); his tools only read
transcripts and relay the user's words into a session through CloudCLI.
"""

import asyncio
import datetime
import json

import httpx

from server import config

TOOLS = [
    {"type": "function", "function": {
        "name": "sessions_overview",
        "description": ("The user's recently active Claude Code sessions "
                        "across all projects: id, project, how long ago, "
                        "last message."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "read_session",
        "description": ("Read the last few messages of one session, e.g. to "
                        "quote what it said or check what it needs."),
        "parameters": {"type": "object", "properties": {
            "session": {"type": "string", "description":
                        "session id, id prefix, or project name"},
            "count": {"type": "integer"}}, "required": ["session"]},
    }},
    {"type": "function", "function": {
        "name": "send_to_session",
        "description": ("Deliver the user's answer or instruction into a "
                        "Claude Code session. The session treats it as its "
                        "next prompt and its result is announced "
                        "automatically when done."),
        "parameters": {"type": "object", "properties": {
            "session": {"type": "string", "description":
                        "session id, id prefix, or project name"},
            "text": {"type": "string"}}, "required": ["session", "text"]},
    }},
    {"type": "function", "function": {
        "name": "remember",
        "description": ("Store a durable fact about the user, their projects "
                        "or preferences, for all future conversations."),
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string"}}, "required": ["fact"]},
    }},
]

SYSTEM_TEMPLATE = """You are Jarvis, the user's local voice assistant with a 3D face, \
running on their machine. The user works in many Claude Code sessions at \
once; you are their ears and voice over all of them. When any session \
finishes a turn you are notified automatically and you announce it. The \
user answers you by voice and you pass their words into the right session.

Hard rules:
- You never write code, diffs or shell commands yourself. The Claude Code \
sessions do the real work; your value is relaying faithfully and \
summarizing honestly.
- Delivering words into a session is PHYSICALLY IMPOSSIBLE without \
calling send_to_session — there is no other mechanism. When the user says \
"tell them yes", "say go ahead", "ask it to also update the docs", you \
MUST call send_to_session in that same turn, every time, even if a \
similar exchange appears in the recent conversation. The most recently announced session is the target unless \
they name another. If it is genuinely ambiguous, ask which session.
- Relay content faithfully: rephrase for speech, never invent. If the user \
asks what a session said, use read_session and quote the substance.
- When asked what's going on, use sessions_overview — never guess status.
- NEVER claim a message was sent unless you called send_to_session THIS \
turn and it returned "delivered". If it returned an error or "UNCERTAIN", \
tell the user its exact verdict instead. Saying "sent" without having done \
it is the worst failure you can make.
- Use remember for durable facts worth keeping.

Recently announced session turns (newest last):
{sessions}

Speaking style: your reply is spoken aloud by TTS. One to three short natural \
sentences, plain text only — no markdown, no emoji, no lists, no code.

Current date/time: {now}

Long-term facts you know:
{facts}

Summary of earlier conversation:
{summary}
"""

SUMMARIZE_EVERY = 30


class Brain:

  def __init__(self, memory, sessions):
    self.memory = memory
    self.sessions = sessions
    self.backend = config.BRAIN_BACKEND
    self.model = config.OPENROUTER_MODEL
    self.ollama_model = None
    self.client = httpx.AsyncClient(timeout=300)
    self._think_supported = True
    self.tools_ran = []

  async def start(self):
    try:
      tags = (await self.client.get(f"{config.OLLAMA_URL}/api/tags")).json()
      installed = {m["name"] for m in tags.get("models", [])}
      for cand in config.MODEL_CANDIDATES:
        if cand in installed or cand.split(":")[0] in {
            n.split(":")[0] for n in installed}:
          self.ollama_model = cand
          break
    except Exception:  # noqa: BLE001 - ollama optional when openrouter is up
      pass
    if self.backend == "openrouter" and not config.OPENROUTER_KEY:
      self.backend = "ollama"
    if self.backend == "ollama":
      if not self.ollama_model:
        raise RuntimeError("No brain available: no OpenRouter key and no "
                           "ollama model installed.")
      self.model = self.ollama_model
    return f"{self.backend}:{self.model}"

  # -- context assembly --

  def _system(self) -> str:
    facts = self.memory.facts()
    recent = [{"session_id": r["session_id"], "project": r["project"],
               "minutes_ago": int((datetime.datetime.now().timestamp()
                                   - r["at"]) / 60),
               "said": (r.get("last_text") or "")[:200]}
              for r in self.sessions.recent]
    return SYSTEM_TEMPLATE.format(
        now=datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M"),
        facts="\n".join(f"- {f}" for f in facts) or "(none yet)",
        summary=self.memory.get_kv("summary", "(no earlier conversation)"),
        sessions=json.dumps(recent) if recent else "(none yet this run)",
    )

  def _messages(self, user_text: str) -> list[dict]:
    msgs = [{"role": "system", "content": self._system()}]
    for m in self.memory.recent_messages(14):
      msgs.append({"role": m["role"], "content": m["text"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs

  # -- OpenRouter (OpenAI-compatible) --

  def _or_headers(self):
    return {"Authorization": f"Bearer {config.OPENROUTER_KEY}",
            "HTTP-Referer": "http://127.0.0.1:8710",
            "X-Title": "Jarvis"}

  async def _or_stream(self, messages, tools):
    """Yields ('delta', text) then ('done', message) in OpenAI format."""
    payload = {"model": self.model, "messages": messages, "stream": True,
               "temperature": 0.3}
    if tools:
      payload["tools"] = tools
    content, tool_calls = "", {}
    async with self.client.stream(
        "POST", f"{config.OPENROUTER_URL}/chat/completions",
        json=payload, headers=self._or_headers()) as r:
      if r.status_code != 200:
        body = (await r.aread()).decode()[:400]
        raise RuntimeError(f"OpenRouter {r.status_code}: {body}")
      async for line in r.aiter_lines():
        if not line.startswith("data: "):
          continue
        data = line[6:].strip()
        if data == "[DONE]":
          break
        chunk = json.loads(data)
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
          content += delta["content"]
          yield ("delta", delta["content"])
        for tc in delta.get("tool_calls") or []:
          slot = tool_calls.setdefault(tc.get("index", 0),
                                       {"id": None, "name": "", "args": ""})
          if tc.get("id"):
            slot["id"] = tc["id"]
          fn = tc.get("function") or {}
          if fn.get("name"):
            slot["name"] += fn["name"]
          if fn.get("arguments"):
            slot["args"] += fn["arguments"]
    calls = []
    for i in sorted(tool_calls):
      slot = tool_calls[i]
      try:
        args = json.loads(slot["args"] or "{}")
      except json.JSONDecodeError:
        args = {}
      calls.append({"id": slot["id"] or f"call_{i}", "type": "function",
                    "function": {"name": slot["name"],
                                 "arguments": slot["args"] or "{}"},
                    "_parsed_args": args})
    yield ("done", {"role": "assistant", "content": content,
                    "tool_calls": calls})

  async def _turn_openai(self, messages):
    for _round in range(6):
      final = None
      async for kind, data in self._or_stream(messages, TOOLS):
        if kind == "delta":
          yield ("text", data)
        else:
          final = data
      if not final["tool_calls"]:
        return
      messages.append({"role": "assistant",
                       "content": final["content"] or None,
                       "tool_calls": [{k: v for k, v in tc.items()
                                       if k != "_parsed_args"}
                                      for tc in final["tool_calls"]]})
      for tc in final["tool_calls"]:
        result = await self._run_tool(tc["function"]["name"],
                                      tc["_parsed_args"])
        messages.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": result})

  # -- ollama --

  async def _ollama_stream(self, messages, tools, model):
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"temperature": 0.3}}
    if tools:
      payload["tools"] = tools
    if self._think_supported:
      payload["think"] = False
    content, tool_calls = "", []
    async with self.client.stream(
        "POST", f"{config.OLLAMA_URL}/api/chat", json=payload) as r:
      if r.status_code == 400 and self._think_supported:
        self._think_supported = False
        async for ev in self._ollama_stream(messages, tools, model):
          yield ev
        return
      r.raise_for_status()
      async for line in r.aiter_lines():
        if not line.strip():
          continue
        chunk = json.loads(line)
        msg = chunk.get("message") or {}
        if msg.get("content"):
          content += msg["content"]
          yield ("delta", msg["content"])
        tool_calls += msg.get("tool_calls") or []
        if chunk.get("done"):
          break
    yield ("done", {"role": "assistant", "content": content,
                    "tool_calls": tool_calls})

  async def _turn_ollama(self, messages):
    model = self.ollama_model
    if not model:
      raise RuntimeError("no local model available")
    for _round in range(6):
      final = None
      async for kind, data in self._ollama_stream(messages, TOOLS, model):
        if kind == "delta":
          yield ("text", data)
        else:
          final = data
      if not final["tool_calls"]:
        return
      messages.append(final)
      for tc in final["tool_calls"]:
        fn = tc.get("function") or {}
        result = await self._run_tool(fn.get("name", "?"),
                                      fn.get("arguments") or {})
        messages.append({"role": "tool", "content": result,
                         "tool_name": fn.get("name", "?")})

  # -- tools --

  async def _run_tool(self, name: str, args: dict) -> str:
    self.tools_ran.append(name)
    print(f"[brain] tool call: {name} {str(args)[:120]}", flush=True)
    try:
      if name == "sessions_overview":
        return json.dumps(self.sessions.overview() or "no recent sessions")
      if name == "read_session":
        msgs = self.sessions.read_session(args["session"],
                                          int(args.get("count", 6)))
        return json.dumps(msgs or "session not found or empty")
      if name == "send_to_session":
        return await self.sessions.send(args["session"], args["text"])
      if name == "remember":
        self.memory.add_fact(args["fact"], source="jarvis")
        return "remembered"
      return f"unknown tool {name}"
    except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
      return f"tool error: {exc}"

  # -- public API --

  async def chat(self, user_text: str):
    """Async generator yielding ('text', delta) for the spoken reply."""
    self.tools_ran = []
    self.memory.add_message("user", user_text)
    messages = self._messages(user_text)
    full = []

    async def consume(gen):
      async for kind, delta in gen:
        full.append(delta)
        yield (kind, delta)

    try:
      if self.backend == "openrouter":
        async for ev in consume(self._turn_openai(messages)):
          yield ev
      else:
        async for ev in consume(self._turn_ollama(messages)):
          yield ev
    except (httpx.HTTPError, RuntimeError) as exc:
      # Cloud brain unreachable -> retry the whole turn locally.
      if self.backend == "openrouter" and self.ollama_model and not full:
        async for ev in consume(self._turn_ollama(self._messages(user_text))):
          yield ev
      else:
        raise exc

    reply = "".join(full).strip()
    if reply:
      self.memory.add_message("assistant", reply)
    asyncio.get_running_loop().create_task(self._maybe_summarize())

  async def _oneshot(self, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    parts = []
    try:
      if self.backend == "openrouter":
        async for kind, data in self._or_stream(messages, None):
          if kind == "delta":
            parts.append(data)
      else:
        async for kind, data in self._ollama_stream(
            messages, None, self.ollama_model):
          if kind == "delta":
            parts.append(data)
    except (httpx.HTTPError, RuntimeError):
      if self.backend == "openrouter" and self.ollama_model:
        async for kind, data in self._ollama_stream(
            messages, None, self.ollama_model):
          if kind == "delta":
            parts.append(data)
    return "".join(parts).strip()

  async def announce_turn(self, info: dict) -> str:
    """One short spoken update for a finished session turn."""
    said = (info.get("last_text") or "").strip()
    asked = (info.get("last_user") or "").strip()
    context = (f"The user had said:\n{asked[:400]}\n\n" if asked else "")
    prompt = (
        f"The Claude Code session in project '{info['project']}' just "
        f"finished a turn. {context}"
        f"Its message to the user:\n{said[:1500]}\n\n"
        "Compose ONE or TWO short spoken sentences relaying this, plain "
        "text, natural, mention the project. If it asks the user something, "
        "make the question clear so they can answer by voice. Reply with "
        "only those sentences.")
    text = await self._oneshot(prompt)
    announcement = text or (
        f"The {info['project']} session just finished a turn.")
    self.memory.add_message(
        "assistant", f"[{info['project']} session] {announcement}")
    return announcement

  async def _maybe_summarize(self):
    last_id = self.memory.last_message_id()
    done_to = int(self.memory.get_kv("summarized_to", "0"))
    if last_id - done_to < SUMMARIZE_EVERY:
      return
    cutoff = last_id - 14
    if cutoff <= done_to:
      return
    older = self.memory.messages_range(done_to, cutoff)
    if not older:
      return
    convo = "\n".join(f"{m['role']}: {m['text']}" for m in older)
    prev = self.memory.get_kv("summary", "")
    prompt = (
        "Update this running summary of a conversation between a user and "
        f"their assistant Jarvis.\n\nCurrent summary:\n{prev or '(empty)'}\n\n"
        f"New messages:\n{convo}\n\nReply with only the updated summary, "
        "under 200 words, keeping decisions, ongoing work and stable facts.")
    try:
      summary = await self._oneshot(prompt)
      if summary:
        self.memory.set_kv("summary", summary)
        self.memory.set_kv("summarized_to", str(cutoff))
    except Exception:  # noqa: BLE001 - summarization is best-effort
      pass
