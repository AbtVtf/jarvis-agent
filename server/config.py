"""Central configuration (env-overridable)."""

import json
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

def _cloudcli_conf():
  path = os.path.join(ROOT, "data", "cloudcli.json")
  try:
    return json.load(open(path))
  except (OSError, ValueError):
    return {}


_CLOUDCLI = _cloudcli_conf()
CLOUDCLI_URL = os.environ.get(
    "JARVIS_CLOUDCLI_URL", _CLOUDCLI.get("url", "http://127.0.0.1:3001"))
CLOUDCLI_USER = os.environ.get(
    "JARVIS_CLOUDCLI_USER", _CLOUDCLI.get("username", ""))
CLOUDCLI_PASS = os.environ.get(
    "JARVIS_CLOUDCLI_PASS", _CLOUDCLI.get("password", ""))


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


STT_MODEL = os.environ.get("JARVIS_STT_MODEL", "distil-large-v3")
WAKE_WORD = os.environ.get("JARVIS_WAKE_WORD", "hey_jarvis")
WAKE_THRESHOLD = float(os.environ.get("JARVIS_WAKE_THRESHOLD", "0.5"))

HOST = "127.0.0.1"
PORT = int(os.environ.get("JARVIS_PORT", "8710"))
