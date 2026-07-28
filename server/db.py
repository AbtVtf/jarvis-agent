"""Jarvis memory: SQLite-backed conversation history, facts, and agent runs.

The brain's context is assembled from: pinned facts + rolling summary of old
conversation + recent messages verbatim + live agent snapshot.
"""

import os
import sqlite3
import threading
import time

from server import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  role TEXT NOT NULL,
  text TEXT NOT NULL,
  ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  source TEXT,
  ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_runs (
  id TEXT PRIMARY KEY,
  project TEXT NOT NULL,
  path TEXT NOT NULL,
  task TEXT NOT NULL,
  status TEXT NOT NULL,
  session_id TEXT,
  last_activity TEXT,
  result TEXT,
  created_ts REAL NOT NULL,
  updated_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
  key TEXT PRIMARY KEY,
  value TEXT
);
CREATE TABLE IF NOT EXISTS reminders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  due_ts REAL NOT NULL,
  fired INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS todos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  done INTEGER DEFAULT 0,
  created_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  created_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS schedules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hhmm TEXT NOT NULL,
  weekdays TEXT DEFAULT 'daily',
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,
  last_run_date TEXT DEFAULT ''
);
"""


class Memory:

  def __init__(self, path: str = None):
    path = path or config.DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    self._lock = threading.Lock()
    self._conn = sqlite3.connect(path, check_same_thread=False)
    self._conn.row_factory = sqlite3.Row
    self._conn.execute("PRAGMA journal_mode=WAL")
    self._conn.executescript(_SCHEMA)
    self._conn.commit()

  def _exec(self, sql, args=()):
    with self._lock:
      cur = self._conn.execute(sql, args)
      self._conn.commit()
      return cur

  # -- conversation --

  def add_message(self, role: str, text: str):
    self._exec("INSERT INTO messages (role, text, ts) VALUES (?,?,?)",
               (role, text, time.time()))

  def recent_messages(self, n: int = 16):
    rows = self._exec(
        "SELECT role, text FROM messages ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    return [dict(r) for r in reversed(rows)]

  def message_count(self) -> int:
    return self._exec("SELECT COUNT(*) c FROM messages").fetchone()["c"]

  def messages_range(self, start_id: int, end_id: int):
    rows = self._exec(
        "SELECT id, role, text FROM messages WHERE id > ? AND id <= ? ORDER BY id",
        (start_id, end_id)).fetchall()
    return [dict(r) for r in rows]

  def last_message_id(self) -> int:
    row = self._exec("SELECT MAX(id) m FROM messages").fetchone()
    return row["m"] or 0

  # -- rolling summary --

  def get_kv(self, key: str, default: str = "") -> str:
    row = self._exec("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default

  def set_kv(self, key: str, value: str):
    self._exec("INSERT INTO kv (key, value) VALUES (?,?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
               (key, value))

  # -- facts --

  def add_fact(self, text: str, source: str = "user"):
    self._exec("INSERT INTO facts (text, source, ts) VALUES (?,?,?)",
               (text, source, time.time()))

  def facts(self, limit: int = 40):
    rows = self._exec(
        "SELECT text FROM facts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [r["text"] for r in reversed(rows)]

  # -- agent runs --

  def upsert_agent_run(self, run: dict):
    self._exec(
        """INSERT INTO agent_runs
           (id, project, path, task, status, session_id, last_activity,
            result, created_ts, updated_ts)
           VALUES (:id,:project,:path,:task,:status,:session_id,
                   :last_activity,:result,:created_ts,:updated_ts)
           ON CONFLICT(id) DO UPDATE SET
             status=excluded.status, session_id=excluded.session_id,
             last_activity=excluded.last_activity, result=excluded.result,
             updated_ts=excluded.updated_ts""",
        run)

  def agent_runs(self, limit: int = 30):
    rows = self._exec(
        "SELECT * FROM agent_runs ORDER BY updated_ts DESC LIMIT ?",
        (limit,)).fetchall()
    return [dict(r) for r in rows]
