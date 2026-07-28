"""Claude Code agent orchestration.

Spawns headless `claude -p` sessions per project with stream-json output,
parses the event stream live (current tool activity, final result), supports
resuming a session with follow-up prompts, and emits events upward so Jarvis
can announce completions.
"""

import asyncio
import collections
import json
import os
import time
import uuid
from typing import Awaitable, Callable

from server import config

Event = Callable[["AgentRun", str], Awaitable[None]]  # (run, kind)


def _summarize_tool_use(name: str, tool_input: dict) -> str:
  target = (tool_input.get("file_path") or tool_input.get("path")
            or tool_input.get("pattern") or tool_input.get("command")
            or tool_input.get("url") or "")
  target = str(target)
  if len(target) > 70:
    target = target[:67] + "..."
  return f"{name}: {target}" if target else name


class AgentRun:

  def __init__(self, project: str, path: str, task: str, full_auto: bool):
    self.id = uuid.uuid4().hex[:8]
    self.project = project
    self.path = path
    self.task = task
    self.full_auto = full_auto
    self.status = "starting"  # starting|running|done|error
    self.session_id = None
    self.last_activity = "starting up"
    self.activity_log = collections.deque(maxlen=80)
    self.result = None
    self.num_turns = 0
    self.created_ts = time.time()
    self.updated_ts = time.time()
    self.finished_ts = None
    self._proc = None

  def to_row(self) -> dict:
    return {
        "id": self.id, "project": self.project, "path": self.path,
        "task": self.task, "status": self.status,
        "session_id": self.session_id, "last_activity": self.last_activity,
        "result": self.result, "created_ts": self.created_ts,
        "updated_ts": self.updated_ts,
    }

  def brief(self) -> dict:
    elapsed = int((self.finished_ts or time.time()) - self.created_ts)
    return {
        "id": self.id, "project": self.project,
        "task": self.task if len(self.task) < 140 else self.task[:137] + "...",
        "status": self.status, "last_activity": self.last_activity,
        "elapsed_s": elapsed,
        "finished_ago_s": (int(time.time() - self.finished_ts)
                           if self.finished_ts else None),
    }


class Orchestrator:

  def __init__(self, memory, on_event: Event):
    self.memory = memory
    self.on_event = on_event
    self.runs: dict[str, AgentRun] = {}

  # -- projects --

  def list_projects(self) -> list[dict]:
    projects = []
    root = config.PROJECTS_ROOT
    try:
      entries = sorted(os.listdir(root))
    except FileNotFoundError:
      return []
    for name in entries:
      path = os.path.join(root, name)
      if not os.path.isdir(path) or name.startswith("."):
        continue
      projects.append({
          "name": name,
          "path": path,
          "is_git": os.path.isdir(os.path.join(path, ".git")),
          "has_claude_md": os.path.isfile(os.path.join(path, "CLAUDE.md")),
      })
    return projects

  def resolve_project(self, name: str) -> dict | None:
    name_l = name.lower().strip()
    projects = self.list_projects()
    for p in projects:
      if p["name"].lower() == name_l:
        return p
    for p in projects:
      if name_l in p["name"].lower():
        return p
    return None

  # -- lifecycle --

  def active_count(self) -> int:
    return sum(1 for r in self.runs.values()
               if r.status in ("starting", "running"))

  async def spawn(self, project: str, task: str, full_auto: bool = False,
                  max_budget_usd: float | None = None) -> AgentRun | str:
    """Returns the AgentRun, or an error string."""
    proj = self.resolve_project(project)
    if not proj:
      names = ", ".join(p["name"] for p in self.list_projects()[:25])
      return f"Unknown project '{project}'. Available: {names}"
    if self.active_count() >= config.MAX_CONCURRENT_AGENTS:
      return (f"Already running {self.active_count()} agents; "
              "wait for one to finish.")

    run = AgentRun(proj["name"], proj["path"], task, full_auto)
    self.runs[run.id] = run
    cmd = self._base_cmd(run) + ["-p", task]
    if max_budget_usd:
      cmd += ["--max-budget-usd", str(max_budget_usd)]
    await self._launch(run, cmd)
    return run

  async def send(self, run_id: str, message: str) -> str:
    run = self.runs.get(run_id)
    if not run:
      return f"No agent with id {run_id}."
    if run.status in ("starting", "running"):
      return (f"Agent {run_id} is still working "
              f"(currently: {run.last_activity}). Wait for it to finish.")
    if not run.session_id:
      return f"Agent {run_id} has no resumable session."
    run.status = "starting"
    run.result = None
    run.last_activity = "resuming with follow-up"
    cmd = self._base_cmd(run) + ["--resume", run.session_id, "-p", message]
    await self._launch(run, cmd)
    return f"Follow-up sent to agent {run_id}; it is working on it now."

  def _base_cmd(self, run: AgentRun) -> list[str]:
    cmd = [config.CLAUDE_BIN, "--output-format", "stream-json", "--verbose"]
    if run.full_auto:
      cmd.append("--dangerously-skip-permissions")
    else:
      cmd += ["--permission-mode", config.AGENT_PERMISSION_MODE]
    return cmd

  async def _launch(self, run: AgentRun, cmd: list[str]):
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    run._proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=run.path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env)
    asyncio.get_running_loop().create_task(self._watch(run))

  async def _watch(self, run: AgentRun):
    run.status = "running"
    self._persist(run)
    await self.on_event(run, "started")
    try:
      async for raw in run._proc.stdout:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
          continue
        try:
          event = json.loads(line)
        except json.JSONDecodeError:
          continue
        self._handle_event(run, event)
      await run._proc.wait()
      if run.status != "done":
        stderr = (await run._proc.stderr.read()).decode("utf-8", "replace")
        run.status = "error"
        run.result = run.result or stderr[-800:] or "process exited abnormally"
    except Exception as exc:  # noqa: BLE001 - keep the orchestrator alive
      run.status = "error"
      run.result = f"orchestrator error: {exc}"
    run.finished_ts = time.time()
    run.updated_ts = time.time()
    self._persist(run)
    await self.on_event(run, "finished")

  def _handle_event(self, run: AgentRun, event: dict):
    etype = event.get("type")
    if etype == "system" and event.get("subtype") == "init":
      run.session_id = event.get("session_id", run.session_id)
    elif etype == "assistant":
      content = (event.get("message") or {}).get("content") or []
      for block in content:
        if block.get("type") == "tool_use":
          activity = _summarize_tool_use(
              block.get("name", "?"), block.get("input") or {})
          run.last_activity = activity
          run.activity_log.append(f"[{time.strftime('%H:%M:%S')}] {activity}")
        elif block.get("type") == "text" and block.get("text", "").strip():
          snippet = block["text"].strip().replace("\n", " ")
          if len(snippet) > 90:
            snippet = snippet[:87] + "..."
          run.activity_log.append(f"[{time.strftime('%H:%M:%S')}] 💬 {snippet}")
    elif etype == "result":
      run.session_id = event.get("session_id", run.session_id)
      run.num_turns = event.get("num_turns", 0)
      run.result = event.get("result") or event.get("error") or ""
      denials = event.get("permission_denials") or []
      if denials:
        denied = ", ".join(sorted({d.get("tool_name", "?") for d in denials}))
        run.result += (f"\n[note: {len(denials)} action(s) were blocked by "
                       f"permissions ({denied}); work may be incomplete]")
      run.status = "error" if event.get("is_error") else "done"
    run.updated_ts = time.time()

  def _persist(self, run: AgentRun):
    self.memory.upsert_agent_run(run.to_row())

  # -- introspection for the brain --

  def snapshot(self, include_done: bool = True) -> list[dict]:
    runs = sorted(self.runs.values(), key=lambda r: r.created_ts)
    out = []
    for r in runs:
      if not include_done and r.status not in ("starting", "running"):
        continue
      out.append(r.brief())
    return out

  def details(self, run_id: str) -> dict | str:
    run = self.runs.get(run_id)
    if not run:
      return f"No agent with id {run_id}."
    d = run.brief()
    d["activity_tail"] = list(run.activity_log)[-15:]
    if run.result:
      d["result"] = run.result[:2000]
    return d
