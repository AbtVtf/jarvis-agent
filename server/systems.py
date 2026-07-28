"""Systems integration: plug external projects into Jarvis via jarvis.md.

Any project under PROJECTS_ROOT becomes a "system" by having a jarvis.md at
its root: YAML-ish frontmatter (name, description, data globs, declared
commands) followed by free-form notes that are given to Jarvis verbatim.

Example jarvis.md:
    ---
    name: meal-planner
    description: Weekly meal basket and grocery price comparison
    data:
      - basket.json
      - recipes/*.md
    commands:
      - name: compare_prices
        run: python3 pricewatch.py compare
        description: Live price table across stores
    ---
    The basket implies the week's dinners. Walk recipes one step at a time.

Security model: Jarvis can only run commands DECLARED in a manifest (written
by the repo owner), always with cwd inside that project, with a timeout.
"""

import asyncio
import glob as globmod
import os
import re

from server import config

MAX_FILE_CHARS = 24_000
CMD_TIMEOUT_S = 180


def _parse_manifest(path: str) -> dict | None:
  try:
    text = open(path, encoding="utf-8", errors="replace").read()
  except OSError:
    return None
  m = re.match(r"\s*---\n(.*?)\n---\n?(.*)", text, re.S)
  if not m:
    return None
  front, notes = m.group(1), m.group(2).strip()
  manifest = {"data": [], "commands": [], "notes": notes,
              "root": os.path.dirname(path)}
  current_cmd = None
  section = None
  for line in front.splitlines():
    if not line.strip() or line.strip().startswith("#"):
      continue
    top = re.match(r"^(\w+):\s*(.*)$", line)
    if top:
      key, val = top.group(1), top.group(2).strip()
      if key in ("name", "description"):
        manifest[key] = val
        section = None
      elif key in ("data", "commands"):
        section = key
      continue
    item = re.match(r"^\s+-\s+(.*)$", line)
    if item and section == "data":
      manifest["data"].append(item.group(1).strip())
    elif item and section == "commands":
      current_cmd = {}
      manifest["commands"].append(current_cmd)
      kv = re.match(r"^(\w+):\s*(.*)$", item.group(1).strip())
      if kv:
        current_cmd[kv.group(1)] = kv.group(2).strip()
    elif section == "commands" and current_cmd is not None:
      kv = re.match(r"^\s+(\w+):\s*(.*)$", line)
      if kv:
        current_cmd[kv.group(1)] = kv.group(2).strip()
  if "name" not in manifest:
    manifest["name"] = os.path.basename(os.path.dirname(path))
  return manifest


class Systems:

  def discover(self) -> list[dict]:
    systems = []
    root = config.PROJECTS_ROOT
    try:
      entries = sorted(os.listdir(root))
    except FileNotFoundError:
      return []
    for name in entries:
      mpath = os.path.join(root, name, "jarvis.md")
      if os.path.isfile(mpath):
        man = _parse_manifest(mpath)
        if man:
          systems.append(man)
    return systems

  def get(self, name: str) -> dict | None:
    name_l = name.lower().strip()
    for man in self.discover():
      if man["name"].lower() == name_l:
        return man
    for man in self.discover():
      if name_l in man["name"].lower():
        return man
    return None

  def overview(self) -> list[dict]:
    return [{"name": m["name"], "description": m.get("description", "")}
            for m in self.discover()]

  def describe(self, name: str) -> dict | str:
    man = self.get(name)
    if not man:
      return f"No system named '{name}'. Known: " + ", ".join(
          m["name"] for m in self.discover())
    files = []
    for pattern in man["data"]:
      matches = globmod.glob(os.path.join(man["root"], pattern))
      files += [os.path.relpath(f, man["root"]) for f in matches[:20]]
    return {
        "name": man["name"],
        "description": man.get("description", ""),
        "notes": man["notes"][:4000],
        "data_files": files,
        "commands": [{"name": c.get("name"),
                      "description": c.get("description", "")}
                     for c in man["commands"] if c.get("name")],
    }

  def read_file(self, name: str, rel_path: str) -> str:
    man = self.get(name)
    if not man:
      return f"No system named '{name}'."
    full = os.path.realpath(os.path.join(man["root"], rel_path))
    if not full.startswith(os.path.realpath(man["root"]) + os.sep):
      return "Path escapes the system's folder; refused."
    # Only files matched by the manifest's data globs (or the README) are
    # readable — the manifest defines the contract.
    allowed = {os.path.realpath(p) for pattern in man["data"] + ["README.md"]
               for p in globmod.glob(os.path.join(man["root"], pattern))}
    if full not in allowed:
      return ("That file is not exposed by the system's jarvis.md data list. "
              "Exposed: " + ", ".join(
                  os.path.relpath(p, man["root"]) for p in sorted(allowed)))
    try:
      content = open(full, encoding="utf-8", errors="replace").read()
    except OSError as exc:
      return f"Could not read {rel_path}: {exc}"
    if len(content) > MAX_FILE_CHARS:
      content = content[:MAX_FILE_CHARS] + "\n...[truncated]"
    return content

  async def run_command(self, name: str, command_name: str) -> str:
    man = self.get(name)
    if not man:
      return f"No system named '{name}'."
    cmd = next((c for c in man["commands"]
                if c.get("name") == command_name and c.get("run")), None)
    if not cmd:
      names = ", ".join(c.get("name", "?") for c in man["commands"])
      return f"System '{man['name']}' declares no command '{command_name}'. " \
             f"Available: {names or '(none)'}"
    proc = await asyncio.create_subprocess_shell(
        cmd["run"], cwd=man["root"],
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
      out, _ = await asyncio.wait_for(proc.communicate(), CMD_TIMEOUT_S)
    except asyncio.TimeoutError:
      proc.kill()
      return f"Command '{command_name}' timed out after {CMD_TIMEOUT_S}s."
    text = out.decode("utf-8", "replace")
    if len(text) > MAX_FILE_CHARS:
      text = text[:MAX_FILE_CHARS] + "\n...[truncated]"
    return text or "(command produced no output)"
