"""Daily-life abilities: reminders, todos, notes, weather, briefing, repo
sweep, system health, tech news, media/volume, clipboard, schedules.

Everything here is token-free: Open-Meteo and Hacker News need no keys;
media/volume/clipboard use wpctl / gdbus (MPRIS) / xclip locally.
"""

import asyncio
import datetime
import json
import os
import subprocess
import time

import httpx
import psutil

from server import config

WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
       45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
       55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       66: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
       80: "rain showers", 81: "rain showers", 82: "violent showers",
       95: "thunderstorm", 96: "thunderstorm w/ hail", 99: "thunderstorm"}

DEFAULT_LOCATION = {"name": "Bucharest", "lat": 44.4268, "lon": 26.1025}


def _run(cmd: list[str], timeout=10) -> str:
  try:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (out.stdout + out.stderr).strip()
  except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
    return f"error: {exc}"


class Daily:

  def __init__(self, memory, orch):
    self.memory = memory
    self.orch = orch
    self.http = httpx.AsyncClient(timeout=25)

  # ---- reminders ----

  def set_reminder(self, text: str, due_iso: str) -> str:
    try:
      due = datetime.datetime.fromisoformat(due_iso)
    except ValueError:
      return f"Could not parse '{due_iso}' as an ISO datetime."
    if due.timestamp() < time.time() - 60:
      return "That time is in the past."
    self.memory._exec("INSERT INTO reminders (text, due_ts) VALUES (?,?)",
                      (text, due.timestamp()))
    return f"Reminder set for {due.strftime('%H:%M on %A')}: {text}"

  def list_reminders(self) -> list[dict]:
    rows = self.memory._exec(
        "SELECT id, text, due_ts FROM reminders WHERE fired=0 "
        "ORDER BY due_ts LIMIT 20").fetchall()
    return [{"id": r["id"], "text": r["text"],
             "due": datetime.datetime.fromtimestamp(r["due_ts"])
             .strftime("%Y-%m-%d %H:%M")} for r in rows]

  def cancel_reminder(self, reminder_id: int) -> str:
    self.memory._exec("UPDATE reminders SET fired=1 WHERE id=?",
                      (reminder_id,))
    return "cancelled"

  def due_reminders(self) -> list[dict]:
    rows = self.memory._exec(
        "SELECT id, text FROM reminders WHERE fired=0 AND due_ts<=?",
        (time.time(),)).fetchall()
    for r in rows:
      self.memory._exec("UPDATE reminders SET fired=1 WHERE id=?", (r["id"],))
    return [dict(r) for r in rows]

  # ---- todos ----

  def add_todo(self, text: str) -> str:
    self.memory._exec("INSERT INTO todos (text, created_ts) VALUES (?,?)",
                      (text, time.time()))
    return "added"

  def list_todos(self, include_done=False) -> list[dict]:
    q = "SELECT id, text, done FROM todos"
    if not include_done:
      q += " WHERE done=0"
    rows = self.memory._exec(q + " ORDER BY id DESC LIMIT 30").fetchall()
    return [dict(r) for r in rows]

  def complete_todo(self, query: str) -> str:
    rows = self.memory._exec(
        "SELECT id, text FROM todos WHERE done=0").fetchall()
    if query.isdigit():
      match = next((r for r in rows if r["id"] == int(query)), None)
    else:
      match = next((r for r in rows if query.lower() in r["text"].lower()),
                   None)
    if not match:
      return f"No open todo matching '{query}'."
    self.memory._exec("UPDATE todos SET done=1 WHERE id=?", (match["id"],))
    return f"Done: {match['text']}"

  # ---- notes ----

  def add_note(self, text: str) -> str:
    self.memory._exec("INSERT INTO notes (text, created_ts) VALUES (?,?)",
                      (text, time.time()))
    return "noted"

  def search_notes(self, query: str = "") -> list[dict]:
    if query:
      rows = self.memory._exec(
          "SELECT id, text, created_ts FROM notes WHERE text LIKE ? "
          "ORDER BY id DESC LIMIT 12", (f"%{query}%",)).fetchall()
    else:
      rows = self.memory._exec(
          "SELECT id, text, created_ts FROM notes ORDER BY id DESC LIMIT 12"
      ).fetchall()
    return [{"text": r["text"],
             "when": datetime.datetime.fromtimestamp(r["created_ts"])
             .strftime("%b %d")} for r in rows]

  # ---- weather & location ----

  def _location(self) -> dict:
    raw = self.memory.get_kv("location", "")
    return json.loads(raw) if raw else DEFAULT_LOCATION

  async def set_location(self, city: str) -> str:
    r = await self.http.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1})
    hits = r.json().get("results") or []
    if not hits:
      return f"Could not find a place called '{city}'."
    hit = hits[0]
    loc = {"name": hit["name"], "lat": hit["latitude"], "lon": hit["longitude"]}
    self.memory.set_kv("location", json.dumps(loc))
    return f"Location set to {hit['name']}, {hit.get('country', '')}."

  async def weather(self) -> dict:
    loc = self._location()
    r = await self.http.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": loc["lat"], "longitude": loc["lon"],
                "current": "temperature_2m,apparent_temperature,weather_code,"
                           "wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,"
                         "precipitation_probability_max,weather_code",
                "forecast_days": 3, "timezone": "auto"})
    d = r.json()
    cur, day = d.get("current", {}), d.get("daily", {})
    out = {"location": loc["name"],
           "now": {"temp_c": cur.get("temperature_2m"),
                   "feels_c": cur.get("apparent_temperature"),
                   "sky": WMO.get(cur.get("weather_code"), "unknown"),
                   "wind_kmh": cur.get("wind_speed_10m")},
           "days": []}
    for i, date in enumerate(day.get("time", [])[:3]):
      out["days"].append({
          "date": date,
          "max_c": day["temperature_2m_max"][i],
          "min_c": day["temperature_2m_min"][i],
          "rain_pct": day["precipitation_probability_max"][i],
          "sky": WMO.get(day["weather_code"][i], "unknown")})
    return out

  # ---- repos ----

  def repos_status(self) -> list[dict]:
    out = []
    for p in self.orch.list_projects():
      if not p["is_git"]:
        continue
      path = p["path"]
      dirty = _run(["git", "-C", path, "status", "--porcelain"])
      branch = _run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"])
      last = _run(["git", "-C", path, "log", "-1", "--format=%cr"])
      n_dirty = len([l for l in dirty.splitlines() if l.strip()]) \
          if not dirty.startswith("error") else 0
      if n_dirty or "hour" in last or "minute" in last:
        out.append({"repo": p["name"], "branch": branch,
                    "uncommitted_files": n_dirty, "last_commit": last})
    return out or [{"note": "all repos clean and quiet"}]

  # ---- system health ----

  def system_health(self) -> dict:
    disk = psutil.disk_usage("/")
    mem = psutil.virtual_memory()
    gpu = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total,"
                "temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits"])
    parts = [p.strip() for p in gpu.split(",")] if "," in gpu else []
    return {
        "cpu_pct": psutil.cpu_percent(interval=0.3),
        "ram_used_gb": round(mem.used / 1e9, 1),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "disk_free_gb": round(disk.free / 1e9, 1),
        "disk_used_pct": disk.percent,
        "gpu": {"vram_used_mb": parts[0] if parts else "?",
                "vram_total_mb": parts[1] if len(parts) > 1 else "?",
                "temp_c": parts[2] if len(parts) > 2 else "?",
                "util_pct": parts[3] if len(parts) > 3 else "?"},
    }

  def health_alerts(self) -> list[str]:
    alerts = []
    disk = psutil.disk_usage("/")
    if disk.free < 20e9:
      alerts.append(f"disk space is low: {disk.free/1e9:.0f} gigabytes left")
    gpu = _run(["nvidia-smi", "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits"])
    if gpu.isdigit() and int(gpu) >= 88:
      alerts.append(f"the GPU is running hot at {gpu} degrees")
    return alerts

  # ---- tech news ----

  async def tech_news(self, count: int = 5) -> list[dict]:
    r = await self.http.get("https://hn.algolia.com/api/v1/search",
                            params={"tags": "front_page",
                                    "hitsPerPage": min(count, 10)})
    hits = r.json().get("hits", [])
    return [{"title": h.get("title"), "points": h.get("points"),
             "comments": h.get("num_comments")} for h in hits]

  # ---- media / volume / clipboard ----

  def _mpris_players(self) -> list[str]:
    out = _run(["gdbus", "call", "--session", "--dest",
                "org.freedesktop.DBus", "--object-path",
                "/org/freedesktop/DBus", "--method",
                "org.freedesktop.DBus.ListNames"])
    return [tok.strip("'\" ") for tok in out.replace("(", "").split(",")
            if "org.mpris.MediaPlayer2" in tok]

  def _mpris_status(self, player: str) -> str:
    out = _run(["gdbus", "call", "--session", "--dest", player,
                "--object-path", "/org/mpris/MediaPlayer2", "--method",
                "org.freedesktop.DBus.Properties.Get",
                "org.mpris.MediaPlayer2.Player", "PlaybackStatus"])
    return "Playing" if "Playing" in out else "Paused"

  def media(self, action: str, level_pct: int | None = None) -> str:
    action = action.lower().strip()
    if action in ("volume_up", "volume_down", "mute", "set_volume"):
      if action == "mute":
        _run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"])
        return "toggled mute"
      if action == "set_volume" and level_pct is not None:
        _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@",
              f"{max(0, min(level_pct, 120))}%"])
        return f"volume set to {level_pct}%"
      delta = "10%+" if action == "volume_up" else "10%-"
      _run(["wpctl", "set-volume", "-l", "1.2", "@DEFAULT_AUDIO_SINK@", delta])
      vol = _run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
      return vol or "adjusted"

    players = self._mpris_players()
    if not players:
      return "No media players are running."
    playing = [p for p in players if self._mpris_status(p) == "Playing"]
    target = (playing or players)[0]
    method = {"play": "Play", "pause": "Pause", "playpause": "PlayPause",
              "next": "Next", "previous": "Previous"}.get(action)
    if not method:
      return f"Unknown media action '{action}'."
    _run(["gdbus", "call", "--session", "--dest", target, "--object-path",
          "/org/mpris/MediaPlayer2", "--method",
          f"org.mpris.MediaPlayer2.Player.{method}"])
    nice = target.split("MediaPlayer2.")[-1].split(".")[0]
    return f"{action} sent to {nice}"

  def clipboard(self, action: str, text: str = "") -> str:
    if action == "write":
      try:
        subprocess.run(["xclip", "-selection", "clipboard"],
                       input=text, text=True, timeout=5)
        return "copied to clipboard"
      except Exception as exc:  # noqa: BLE001
        return f"clipboard error: {exc}"
    out = _run(["xclip", "-selection", "clipboard", "-o"])
    return out[:2000] if out else "(clipboard is empty)"

  # ---- schedules ----

  def schedule_task(self, hhmm: str, kind: str, payload: dict,
                    weekdays: str = "daily") -> str:
    if kind not in ("briefing", "say", "agent"):
      return "kind must be one of: briefing, say, agent"
    try:
      datetime.datetime.strptime(hhmm, "%H:%M")
    except ValueError:
      return f"'{hhmm}' is not a valid HH:MM time."
    self.memory._exec(
        "INSERT INTO schedules (hhmm, weekdays, kind, payload) VALUES (?,?,?,?)",
        (hhmm, weekdays.lower(), kind, json.dumps(payload)))
    return f"Scheduled {kind} at {hhmm} ({weekdays})."

  def list_schedules(self) -> list[dict]:
    rows = self.memory._exec(
        "SELECT id, hhmm, weekdays, kind, payload FROM schedules").fetchall()
    return [{"id": r["id"], "at": r["hhmm"], "days": r["weekdays"],
             "kind": r["kind"], "payload": json.loads(r["payload"])}
            for r in rows]

  def cancel_schedule(self, schedule_id: int) -> str:
    self.memory._exec("DELETE FROM schedules WHERE id=?", (schedule_id,))
    return "cancelled"

  def due_schedules(self) -> list[dict]:
    now = datetime.datetime.now()
    hhmm = now.strftime("%H:%M")
    today = now.strftime("%Y-%m-%d")
    dow = now.strftime("%a").lower()
    due = []
    for s in self.list_schedules():
      row = self.memory._exec(
          "SELECT last_run_date FROM schedules WHERE id=?",
          (s["id"],)).fetchone()
      if row["last_run_date"] == today or s["at"] != hhmm:
        continue
      if s["days"] != "daily" and dow not in s["days"]:
        continue
      self.memory._exec("UPDATE schedules SET last_run_date=? WHERE id=?",
                        (today, s["id"]))
      due.append(s)
    return due

  # ---- briefing ----

  async def briefing(self) -> dict:
    try:
      weather = await self.weather()
    except Exception:  # noqa: BLE001
      weather = "unavailable"
    try:
      news = await self.tech_news(4)
    except Exception:  # noqa: BLE001
      news = []
    return {
        "now": datetime.datetime.now().strftime("%A %B %d, %H:%M"),
        "weather": weather,
        "reminders_upcoming": self.list_reminders()[:5],
        "todos_open": self.list_todos()[:6],
        "agents": self.orch.snapshot()[-5:],
        "repos_active": self.repos_status()[:6],
        "tech_news": news,
    }
