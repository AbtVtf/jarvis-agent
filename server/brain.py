"""Jarvis's brain: LLM with orchestration + systems tools.

Backends:
  openrouter (default when a key exists) — OpenAI-compatible streaming with
    tool calls; model set by JARVIS_OPENROUTER_MODEL (deepseek-v4-flash).
  ollama — local fallback (gemma4); also used automatically per-turn when
    OpenRouter is unreachable, so Jarvis survives offline.

Jarvis never writes code himself. His tools spawn/inspect/message real Claude
Code agents, and read/run what integrated systems (jarvis.md manifests)
explicitly expose.
"""

import asyncio
import datetime
import json

import httpx

from server import config

TOOLS = [
    {"type": "function", "function": {
        "name": "list_projects",
        "description": "List the user's project folders that agents can work in.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "spawn_agent",
        "description": (
            "Start a Claude Code agent working on a task inside a project "
            "folder. Write the task as a complete, self-contained brief with "
            "everything the agent needs. Returns the new agent's id."),
        "parameters": {"type": "object", "properties": {
            "project": {"type": "string",
                        "description": "Project folder name (fuzzy matched)"},
            "task": {"type": "string",
                     "description": "Detailed task brief for the agent"},
            "full_auto": {"type": "boolean", "description":
                          "Skip ALL permission checks (only when the user "
                          "explicitly asked for full autonomy)"},
            "max_budget_usd": {"type": "number", "description":
                               "Optional cost cap for this agent run"},
        }, "required": ["project", "task"]},
    }},
    {"type": "function", "function": {
        "name": "agents_overview",
        "description": ("Current status of all agents this session: running, "
                        "done, failed, what each is doing right now."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "agent_details",
        "description": ("Detailed view of one agent: recent activity log and "
                        "its final result if finished."),
        "parameters": {"type": "object", "properties": {
            "agent_id": {"type": "string"}}, "required": ["agent_id"]},
    }},
    {"type": "function", "function": {
        "name": "send_to_agent",
        "description": ("Send a follow-up instruction to a FINISHED agent, "
                        "resuming its session with full context."),
        "parameters": {"type": "object", "properties": {
            "agent_id": {"type": "string"},
            "message": {"type": "string"}},
            "required": ["agent_id", "message"]},
    }},
    {"type": "function", "function": {
        "name": "remember",
        "description": ("Store a durable fact about the user, their projects "
                        "or preferences, for all future conversations."),
        "parameters": {"type": "object", "properties": {
            "fact": {"type": "string"}}, "required": ["fact"]},
    }},
    {"type": "function", "function": {
        "name": "list_systems",
        "description": ("List integrated systems (projects exposing a "
                        "jarvis.md manifest): things like meal planning, "
                        "home data, trackers."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "system_info",
        "description": ("A system's usage notes, exposed data files and "
                        "runnable commands. Call before using a system."),
        "parameters": {"type": "object", "properties": {
            "system": {"type": "string"}}, "required": ["system"]},
    }},
    {"type": "function", "function": {
        "name": "read_system_file",
        "description": "Read one of the data files a system exposes.",
        "parameters": {"type": "object", "properties": {
            "system": {"type": "string"},
            "path": {"type": "string"}}, "required": ["system", "path"]},
    }},
    {"type": "function", "function": {
        "name": "run_system_command",
        "description": ("Run a command DECLARED in a system's manifest. Some "
                        "take a minute — tell the user before running one."),
        "parameters": {"type": "object", "properties": {
            "system": {"type": "string"},
            "command": {"type": "string"}}, "required": ["system", "command"]},
    }},
    {"type": "function", "function": {
        "name": "reminders",
        "description": ("One-off reminders/timers that Jarvis announces out "
                        "loud when due. Compute due_iso yourself from the "
                        "current time (local, ISO like 2026-07-27T18:30)."),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["set", "list", "cancel"]},
            "text": {"type": "string"},
            "due_iso": {"type": "string"},
            "id": {"type": "integer"}}, "required": ["action"]},
    }},
    {"type": "function", "function": {
        "name": "todos",
        "description": "The user's todo list: add items, list open, mark done.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["add", "list", "done"]},
            "text": {"type": "string",
                     "description": "item text; for done: id or fuzzy text"}},
            "required": ["action"]},
    }},
    {"type": "function", "function": {
        "name": "notes",
        "description": ("Quick capture notes ('note that...') and search "
                        "them later. Different from remember: notes are "
                        "content, facts are about the user."),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["add", "search"]},
            "text": {"type": "string"}}, "required": ["action"]},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Current weather + 3-day forecast for the user's city.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "set_location",
        "description": "Change the user's city for weather.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]},
    }},
    {"type": "function", "function": {
        "name": "morning_briefing",
        "description": ("Everything for a spoken briefing: weather, "
                        "reminders, todos, agents, active repos, tech news. "
                        "Narrate the highlights in a few sentences."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "repos_status",
        "description": ("Which git repos have uncommitted changes or very "
                        "recent commits."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "system_health",
        "description": "CPU, RAM, disk and GPU status of this machine.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "tech_news",
        "description": "Top Hacker News front-page stories right now.",
        "parameters": {"type": "object", "properties": {
            "count": {"type": "integer"}}},
    }},
    {"type": "function", "function": {
        "name": "media_control",
        "description": ("Control music/video playback and system volume: "
                        "play, pause, playpause, next, previous, volume_up, "
                        "volume_down, set_volume, mute."),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string"},
            "level_pct": {"type": "integer"}}, "required": ["action"]},
    }},
    {"type": "function", "function": {
        "name": "clipboard",
        "description": "Read the clipboard, or copy dictated text onto it.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["read", "write"]},
            "text": {"type": "string"}}, "required": ["action"]},
    }},
    {"type": "function", "function": {
        "name": "schedules",
        "description": ("Recurring automations at HH:MM local: kind "
                        "'briefing' (spoken briefing), 'say' (speak text), "
                        "'agent' (spawn a Claude Code agent with project+task)."),
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string",
                       "enum": ["add", "list", "cancel"]},
            "hhmm": {"type": "string"},
            "kind": {"type": "string",
                     "enum": ["briefing", "say", "agent"]},
            "weekdays": {"type": "string", "description":
                         "'daily' or e.g. 'mon,tue,wed,thu,fri'"},
            "text": {"type": "string"},
            "project": {"type": "string"},
            "task": {"type": "string"},
            "id": {"type": "integer"}}, "required": ["action"]},
    }},
    {"type": "function", "function": {
        "name": "channels_overview",
        "description": ("External agent sessions connected through Jarvis "
                        "channels: who is active and who awaits a reply."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "send_to_channel",
        "description": ("Deliver the user's message to an external agent "
                        "session waiting on a channel."),
        "parameters": {"type": "object", "properties": {
            "channel": {"type": "string"},
            "text": {"type": "string"}},
            "required": ["channel", "text"]},
    }},
]

SYSTEM_TEMPLATE = """You are Jarvis, the user's local voice assistant with a 3D face, \
running on their machine. You are the orchestration layer over their Claude \
Code agents and their personal systems: you spawn agents to do real work, \
watch them, relay follow-ups, report back, and use integrated systems' data \
to help in daily life.

Hard rules:
- You never write code, diffs or shell commands yourself. Real work is ALWAYS \
delegated to a Claude Code agent via spawn_agent. Your value is judgment: \
picking the right project, writing an excellent task brief, monitoring, and \
summarizing outcomes honestly.
- When the user asks for work: choose the project, then spawn_agent with a \
thorough brief (goal, constraints, how to verify). Confirm briefly. If the \
work creates something Jarvis should later use, tell the agent to write a \
jarvis.md manifest (name, description, data globs, commands) so it becomes \
an integrated system.
- When asked about progress, use agents_overview or agent_details — never \
guess or invent status.
- Integrated systems: when a request touches daily life (meals, tracking, \
home data), check list_systems / system_info and use the exposed data and \
commands. Follow each system's notes.
- Walkthroughs (cooking, assembly, exercises): give exactly ONE short step, \
then wait for the user to say they're done. Never recite everything at once.
- Daily life: you also handle reminders/timers (announced aloud when due), \
todos, notes, weather, spoken briefings, repo status, machine health, tech \
news, media/volume control, the clipboard, and recurring schedules (daily \
briefings or scheduled agent runs). For any reminder or schedule time, \
compute the absolute local time yourself from the current date/time above. \
When briefing, narrate only the interesting parts.
- Use remember for durable facts worth keeping. full_auto only when the user \
explicitly asked for no permission checks.

- Channels: OTHER agent sessions on this machine (interactive Claude Code \
sessions in various repos) message the user through you. Their messages are \
announced automatically. When the user answers one — "tell the padel session \
to go ahead", or just "tell them yes" right after an announcement — deliver \
it with send_to_channel. Check channels_overview when unsure who is waiting; \
if more than one session awaits a reply and the target is unclear, ask which. \
Relay the delivery status truthfully: "delivered" only if the tool said the \
session was listening; if it said QUEUED, tell the user the session isn't \
listening right now and will pick it up when it polls again.

Channel state right now:
{channels}

Speaking style: your reply is spoken aloud by TTS. One to three short natural \
sentences, plain text only — no markdown, no emoji, no lists, no code.

Current date/time: {now}

Long-term facts you know:
{facts}

Summary of earlier conversation:
{summary}

Live agent status right now:
{agents}

Integrated systems available:
{systems}
"""

SUMMARIZE_EVERY = 30


class Brain:

  def __init__(self, memory, orchestrator, systems, daily, channels):
    self.memory = memory
    self.orch = orchestrator
    self.systems = systems
    self.daily = daily
    self.channels = channels
    self.backend = config.BRAIN_BACKEND
    self.model = config.OPENROUTER_MODEL
    self.ollama_model = None
    self.client = httpx.AsyncClient(timeout=300)
    self._think_supported = True

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
    agents = self.orch.snapshot()
    systems = self.systems.overview()
    chans = self.channels.overview()
    return SYSTEM_TEMPLATE.format(
        now=datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M"),
        facts="\n".join(f"- {f}" for f in facts) or "(none yet)",
        summary=self.memory.get_kv("summary", "(no earlier conversation)"),
        agents=json.dumps(agents) if agents else "(no agents yet)",
        systems=json.dumps(systems) if systems else "(none found)",
        channels=json.dumps(chans) if chans else "(no external sessions)",
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
               "temperature": 0.6}
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
               "options": {"temperature": 0.6}}
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
    try:
      if name == "list_projects":
        ps = self.orch.list_projects()
        return json.dumps([{"name": p["name"],
                            "claude_md": p["has_claude_md"]} for p in ps])
      if name == "spawn_agent":
        run = await self.orch.spawn(
            args["project"], args["task"],
            full_auto=bool(args.get("full_auto")),
            max_budget_usd=args.get("max_budget_usd"))
        if isinstance(run, str):
          return run
        return json.dumps({"agent_id": run.id, "project": run.project,
                           "status": run.status})
      if name == "agents_overview":
        return json.dumps(self.orch.snapshot() or "no agents yet")
      if name == "agent_details":
        return json.dumps(self.orch.details(args["agent_id"]))
      if name == "send_to_agent":
        return await self.orch.send(args["agent_id"], args["message"])
      if name == "remember":
        self.memory.add_fact(args["fact"], source="jarvis")
        return "remembered"
      if name == "list_systems":
        return json.dumps(self.systems.overview() or "no systems integrated")
      if name == "system_info":
        return json.dumps(self.systems.describe(args["system"]))
      if name == "read_system_file":
        return self.systems.read_file(args["system"], args["path"])
      if name == "run_system_command":
        return await self.systems.run_command(args["system"], args["command"])
      if name == "reminders":
        act = args["action"]
        if act == "set":
          return self.daily.set_reminder(args["text"], args["due_iso"])
        if act == "cancel":
          return self.daily.cancel_reminder(args["id"])
        return json.dumps(self.daily.list_reminders())
      if name == "todos":
        act = args["action"]
        if act == "add":
          return self.daily.add_todo(args["text"])
        if act == "done":
          return self.daily.complete_todo(args.get("text", ""))
        return json.dumps(self.daily.list_todos())
      if name == "notes":
        if args["action"] == "add":
          return self.daily.add_note(args["text"])
        return json.dumps(self.daily.search_notes(args.get("text", "")))
      if name == "get_weather":
        return json.dumps(await self.daily.weather())
      if name == "set_location":
        return await self.daily.set_location(args["city"])
      if name == "morning_briefing":
        return json.dumps(await self.daily.briefing())
      if name == "repos_status":
        return json.dumps(self.daily.repos_status())
      if name == "system_health":
        return json.dumps(self.daily.system_health())
      if name == "tech_news":
        return json.dumps(await self.daily.tech_news(args.get("count", 5)))
      if name == "media_control":
        return self.daily.media(args["action"], args.get("level_pct"))
      if name == "clipboard":
        return self.daily.clipboard(args["action"], args.get("text", ""))
      if name == "schedules":
        act = args["action"]
        if act == "add":
          kind = args.get("kind", "briefing")
          payload = {}
          if kind == "say":
            payload = {"text": args.get("text", "")}
          elif kind == "agent":
            payload = {"project": args.get("project", ""),
                       "task": args.get("task", "")}
          return self.daily.schedule_task(
              args.get("hhmm", ""), kind, payload,
              args.get("weekdays", "daily"))
        if act == "cancel":
          return self.daily.cancel_schedule(args["id"])
        return json.dumps(self.daily.list_schedules())
      if name == "channels_overview":
        return json.dumps(self.channels.overview() or "no active channels")
      if name == "send_to_channel":
        return self.channels.reply(args["channel"], args["text"])
      return f"unknown tool {name}"
    except Exception as exc:  # noqa: BLE001 - tool errors go back to the model
      return f"tool error: {exc}"

  # -- public API --

  async def chat(self, user_text: str):
    """Async generator yielding ('text', delta) for the spoken reply."""
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

  async def announce(self, run) -> str:
    outcome = (run.result or "")[:600]
    prompt = (
        f"The Claude Code agent working in project '{run.project}' on task "
        f"\"{run.task[:200]}\" just "
        f"{'finished' if run.status == 'done' else 'FAILED'}. "
        f"Its report: {outcome}\n\n"
        "Compose ONE short spoken sentence announcing this to the user, "
        "plain text, natural, mention the project. Reply with only that "
        "sentence.")
    text = await self._oneshot(prompt)
    announcement = text or (
        f"The {run.project} agent just "
        f"{'finished' if run.status == 'done' else 'failed'}.")
    self.memory.add_message("assistant", announcement)
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
