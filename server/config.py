"""Central configuration (env-overridable)."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OLLAMA_URL = os.environ.get("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434")
# Primary brain model; first available from this list wins.
MODEL_CANDIDATES = [
    m for m in [
        os.environ.get("JARVIS_MODEL"),
        "gemma4:12b",
        "qwen3.5:9b",
        "qwen3:14b",
    ] if m
]

PROJECTS_ROOT = os.path.expanduser(
    os.environ.get("JARVIS_PROJECTS_ROOT", "~/Documents/GitHub"))


def _read_key():
  key = os.environ.get("JARVIS_OPENROUTER_KEY", "")
  if key:
    return key
  path = os.path.join(ROOT, "data", "openrouter.key")
  try:
    return open(path).read().strip()
  except OSError:
    return ""


OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY = _read_key()
OPENROUTER_MODEL = os.environ.get("JARVIS_OPENROUTER_MODEL",
                                  "deepseek/deepseek-v4-flash")
BRAIN_BACKEND = os.environ.get(
    "JARVIS_BRAIN", "openrouter" if OPENROUTER_KEY else "ollama")

DB_PATH = os.path.join(ROOT, "data", "jarvis.db")
AUDIO_DIR = os.path.join(ROOT, "build", "audio")
WEB_DIR = os.path.join(ROOT, "web")

CLAUDE_BIN = os.environ.get("JARVIS_CLAUDE_BIN", "claude")
# Default permission mode for spawned agents. "auto" accepts edits and safe
# in-cwd shell; full_auto per-spawn upgrades to --dangerously-skip-permissions
# when the user explicitly asks for it.
AGENT_PERMISSION_MODE = os.environ.get("JARVIS_AGENT_PERMISSIONS", "auto")
MAX_CONCURRENT_AGENTS = int(os.environ.get("JARVIS_MAX_AGENTS", "6"))

STT_MODEL = os.environ.get("JARVIS_STT_MODEL", "distil-large-v3")
WAKE_WORD = os.environ.get("JARVIS_WAKE_WORD", "hey_jarvis")
WAKE_THRESHOLD = float(os.environ.get("JARVIS_WAKE_THRESHOLD", "0.5"))

HOST = "127.0.0.1"
PORT = int(os.environ.get("JARVIS_PORT", "8710"))
