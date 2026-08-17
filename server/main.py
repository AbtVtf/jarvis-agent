"""Jarvis server v3: voice over the user's Claude Code sessions.

Endpoints:
  WS  /ws/ui        JSON protocol with the face UI / widget
  WS  /ws/audio     binary 16k s16le PCM mic stream (wake word + STT)
  POST /claude-event  Claude Code Stop hook: a session finished a turn
  POST /api/chat    REST fallback: full turn, non-streamed
  GET  /            three.js face (web/), /audio/* synthesized clips

UI protocol (server -> client): state, caption_delta, chunk {text,audio,
timeline}, chunks_done, user_text, wake, notify, agents.
Client -> server: {type:'text', text}, {type:'interrupt'}.

Run: .venv/bin/uvicorn server.main:app --host 127.0.0.1 --port 8710
"""

import asyncio
import contextlib
import json
import os
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server import config
from server.brain import Brain
from server.db import Memory
from server.lipsync import LipSync
from server.pipeline import VoicePipeline
from server.sessions import ClaudeSessions
from server.speech import MicSession, SpeechEngine
from server.tts import TTS

state = {}
UI_SOCKETS: set[WebSocket] = set()


async def broadcast(msg: dict):
  dead = []
  for ws in UI_SOCKETS:
    try:
      await ws.send_text(json.dumps(msg))
    except Exception:  # noqa: BLE001
      dead.append(ws)
  for ws in dead:
    UI_SOCKETS.discard(ws)


def _session_cards():
  """Announced turns as cards for the face UI's agent panel."""
  now = time.time()
  return [{"id": r["session_id"][:8], "project": r["project"],
           "status": "done", "task": (r.get("last_text") or "")[:160],
           "elapsed_s": 0, "finished_ago_s": int(now - r["at"])}
          for r in state["sessions"].recent]


async def speak_out(text: str, title: str = "Jarvis", body: str = None):
  """Proactive spoken announcement + desktop notification."""
  await broadcast({"type": "notify", "title": title,
                   "body": (body or text)[:180]})
  async with state["turn_lock"]:
    await broadcast({"type": "state", "state": "speaking"})
    await state["pipeline"].speak_text(text, broadcast)
    await broadcast({"type": "state", "state": "idle"})


async def handle_turn_event(payload: dict):
  info = await state["sessions"].record_turn(
      payload.get("session_id", ""),
      payload.get("cwd", ""),
      payload.get("transcript_path", ""))
  await broadcast({"type": "agents", "agents": _session_cards()})
  if not info["last_text"]:
    return  # nothing visible to announce (tool-only or empty turn)
  try:
    text = await state["brain"].announce_turn(info)
  except Exception:  # noqa: BLE001 - announcement is best-effort
    text = f"The {info['project']} session just finished a turn."
  await speak_out(text, title=f"Jarvis — {info['project']}",
                  body=info["last_text"])


async def run_turn(text: str):
  """One full conversation turn, broadcast to every connected UI."""
  async with state["turn_lock"]:
    await broadcast({"type": "user_text", "text": text})
    await broadcast({"type": "state", "state": "thinking"})
    try:
      reply = await state["pipeline"].speak_deltas(
          state["brain"].chat(text), broadcast)
      if not reply:
        await state["pipeline"].speak_text(
            "Hmm, I came up empty on that one.", broadcast)
    except Exception as exc:  # noqa: BLE001 - keep the server alive
      await broadcast({"type": "error", "text": str(exc)[:300]})
    finally:
      await broadcast({"type": "state", "state": "idle"})


@contextlib.asynccontextmanager
async def lifespan(_app):
  os.makedirs(config.AUDIO_DIR, exist_ok=True)
  state["turn_lock"] = asyncio.Lock()
  state["memory"] = Memory()
  state["tts"] = TTS()
  state["lipsync"] = LipSync()
  state["pipeline"] = VoicePipeline(state["tts"], state["lipsync"])
  state["speech"] = SpeechEngine()
  state["sessions"] = ClaudeSessions()
  state["brain"] = Brain(state["memory"], state["sessions"])
  model = await state["brain"].start()
  state["ready"] = True
  print(f"[jarvis] ready — brain model: {model}")
  yield
  state.clear()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/ui")
async def ws_ui(ws: WebSocket):
  await ws.accept()
  UI_SOCKETS.add(ws)
  await ws.send_text(json.dumps(
      {"type": "agents", "agents": _session_cards()}))
  try:
    while True:
      raw = await ws.receive_text()
      msg = json.loads(raw)
      if msg.get("type") == "text" and msg.get("text", "").strip():
        asyncio.get_running_loop().create_task(run_turn(msg["text"].strip()))
      elif msg.get("type") == "interrupt":
        await broadcast({"type": "state", "state": "idle"})
  except WebSocketDisconnect:
    pass
  finally:
    UI_SOCKETS.discard(ws)


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
  await ws.accept()
  session = MicSession(state["speech"])
  loop = asyncio.get_running_loop()
  try:
    while True:
      packet = await ws.receive()
      if packet.get("type") == "websocket.disconnect":
        break
      events = []
      if packet.get("bytes"):
        events = await loop.run_in_executor(None, session.feed, packet["bytes"])
      elif packet.get("text"):
        cmd = json.loads(packet["text"]).get("cmd")
        if cmd == "capture":  # push-to-talk pressed
          session.set_mode("capture")
          await broadcast({"type": "state", "state": "listening"})
        elif cmd == "finish":  # push-to-talk released
          events = await loop.run_in_executor(None, session.finish)
        elif cmd == "cancel":
          session.set_mode("wake")
          await broadcast({"type": "state", "state": "idle"})
      for event in events:
        if event[0] == "wake":
          await broadcast({"type": "wake"})
          await broadcast({"type": "state", "state": "listening"})
        elif event[0] == "utterance":
          await broadcast({"type": "state", "state": "thinking"})
          loop.create_task(run_turn(event[1]))
        elif event[0] == "timeout":
          await broadcast({"type": "state", "state": "idle"})
  except WebSocketDisconnect:
    pass


@app.post("/claude-event")
async def claude_event(request: Request):
  """Claude Code Stop hook: fire-and-forget, never blocks the session."""
  try:
    payload = await request.json()
  except Exception:  # noqa: BLE001
    return JSONResponse({"error": "bad json"}, status_code=400)
  if not state.get("ready") or not payload.get("session_id"):
    return {"ok": False}
  asyncio.get_running_loop().create_task(handle_turn_event(payload))
  return {"ok": True}


class ChatRequest(BaseModel):
  text: str


@app.post("/api/chat")
async def chat_rest(req: ChatRequest):
  """Non-streamed REST fallback; collects the whole turn."""
  if not req.text.strip():
    return JSONResponse({"error": "empty message"}, status_code=400)
  chunks = []

  async def collect(msg):
    if msg["type"] in ("chunk", "chunks_done"):
      chunks.append(msg)

  async with state["turn_lock"]:
    reply = await state["pipeline"].speak_deltas(
        state["brain"].chat(req.text.strip()), collect)
  return {"reply": reply,
          "chunks": [c for c in chunks if c["type"] == "chunk"]}


@app.get("/api/health")
async def health():
  return {"ok": True, "ready": bool(state.get("ready"))}


app.mount("/audio", StaticFiles(directory=config.AUDIO_DIR), name="audio")
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
