"""Jarvis server v2: voice-first orchestration layer.

Endpoints:
  WS  /ws/ui      JSON protocol with the face UI / widget
  WS  /ws/audio   binary 16k s16le PCM mic stream (wake word + STT)
  POST /api/chat  REST fallback: full turn, non-streamed
  GET  /          three.js face (web/), /audio/* synthesized clips

UI protocol (server -> client): state, caption_delta, chunk {text,audio,
timeline}, chunks_done, user_text, wake, notify, agents.
Client -> server: {type:'text', text}, {type:'interrupt'}.

Run: .venv/bin/uvicorn server.main:app --host 127.0.0.1 --port 8710
"""

import asyncio
import contextlib
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import time

from server import config
from server.brain import Brain
from server.channels import Channels
from server.daily import Daily
from server.db import Memory
from server.lipsync import LipSync
from server.orchestrator import Orchestrator
from server.pipeline import VoicePipeline
from server.speech import MicSession, SpeechEngine
from server.systems import Systems
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


async def on_agent_event(run, kind):
  await broadcast({"type": "agents", "agents": state["orch"].snapshot()})
  if kind == "finished" and state.get("ready"):
    asyncio.get_running_loop().create_task(announce_finish(run))


async def speak_out(text: str, title: str = "Jarvis", body: str = None):
  """Proactive spoken announcement + desktop notification."""
  await broadcast({"type": "notify", "title": title,
                   "body": (body or text)[:180]})
  async with state["turn_lock"]:
    await broadcast({"type": "state", "state": "speaking"})
    await state["pipeline"].speak_text(text, broadcast)
    await broadcast({"type": "state", "state": "idle"})


async def announce_finish(run):
  try:
    text = await state["brain"].announce(run)
  except Exception:  # noqa: BLE001 - announcement is best-effort
    text = f"Heads up: the {run.project} agent just finished."
  await speak_out(text, title=f"Jarvis — {run.project}",
                  body=run.result or text)


async def background_loop():
  last_health_check = 0.0
  while True:
    await asyncio.sleep(20)
    if not state.get("ready"):
      continue
    daily = state["daily"]
    try:
      for r in daily.due_reminders():
        await speak_out(f"Reminder: {r['text']}", title="Jarvis — reminder",
                        body=r["text"])
      for s in daily.due_schedules():
        if s["kind"] == "say":
          await speak_out(s["payload"].get("text", ""))
        elif s["kind"] == "briefing":
          data = await daily.briefing()
          text = await state["brain"]._oneshot(
              "Narrate this briefing to the user in four to six short spoken "
              "sentences, plain text, highlights only:\n" +
              json.dumps(data))
          await speak_out(text or "Here is your briefing.",
                          title="Jarvis — briefing")
        elif s["kind"] == "agent":
          p = s["payload"]
          await state["orch"].spawn(p.get("project", ""), p.get("task", ""))
      if time.time() - last_health_check > 600:
        last_health_check = time.time()
        for alert in daily.health_alerts():
          key = "alerted_" + alert.split(":")[0].replace(" ", "_")[:30]
          if time.time() - float(state["memory"].get_kv(key, "0")) > 7200:
            state["memory"].set_kv(key, str(time.time()))
            await speak_out(f"Heads up: {alert}.", title="Jarvis — system")
    except Exception as exc:  # noqa: BLE001 - the loop must survive anything
      print(f"[background] {exc}")


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
      await broadcast({"type": "agents", "agents": state["orch"].snapshot()})


@contextlib.asynccontextmanager
async def lifespan(_app):
  os.makedirs(config.AUDIO_DIR, exist_ok=True)
  state["turn_lock"] = asyncio.Lock()
  state["memory"] = Memory()
  state["orch"] = Orchestrator(state["memory"], on_agent_event)
  state["tts"] = TTS()
  state["lipsync"] = LipSync()
  state["pipeline"] = VoicePipeline(state["tts"], state["lipsync"])
  state["speech"] = SpeechEngine()
  state["systems"] = Systems()
  state["daily"] = Daily(state["memory"], state["orch"])
  state["channels"] = Channels()
  state["brain"] = Brain(state["memory"], state["orch"], state["systems"],
                         state["daily"], state["channels"])
  model = await state["brain"].start()
  state["ready"] = True
  loop_task = asyncio.get_running_loop().create_task(background_loop())
  print(f"[jarvis] ready — brain model: {model}")
  yield
  loop_task.cancel()
  state.clear()


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/ui")
async def ws_ui(ws: WebSocket):
  await ws.accept()
  UI_SOCKETS.add(ws)
  await ws.send_text(json.dumps(
      {"type": "agents", "agents": state["orch"].snapshot()}))
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


class ChatRequest(BaseModel):
  text: str


class ChannelPost(BaseModel):
  text: str
  expect_reply: bool = False


@app.post("/api/channel/{name}/send")
async def channel_send(name: str, post: ChannelPost):
  """External agents post messages the user hears through Jarvis."""
  text = post.text.strip()
  if not text:
    return JSONResponse({"error": "empty message"}, status_code=400)
  ch = state["channels"].post(name, text, post.expect_reply)
  spoken = text
  if len(spoken) > 400:
    spoken = await state["brain"]._oneshot(
        "Condense this agent report into two short spoken sentences, plain "
        f"text:\n{text[:3000]}") or text[:400]
  suffix = " They're waiting for your answer." if post.expect_reply else ""
  state["memory"].add_message(
      "assistant", f"[{ch.name} session] {text[:500]}")
  asyncio.get_running_loop().create_task(speak_out(
      f"Message from the {ch.name} session: {spoken}{suffix}",
      title=f"Jarvis — {ch.name}", body=text))
  return {"ok": True, "channel": ch.name}


@app.get("/api/channel/{name}/wait")
async def channel_wait(name: str, timeout: float = 300):
  """Long-poll for the user's reply. The reply stays queued until ACKed."""
  reply = await state["channels"].wait_for_reply(name, min(timeout, 570))
  if reply is None:
    return {"timeout": True}
  return reply  # {"id": ..., "reply": ...}


class AckPost(BaseModel):
  id: int


@app.post("/api/channel/{name}/ack")
async def channel_ack(name: str, post: AckPost):
  return {"acked": state["channels"].ack(name, post.id)}


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


@app.get("/api/agents")
async def agents_rest():
  return {"agents": state["orch"].snapshot(),
          "projects": [p["name"] for p in state["orch"].list_projects()]}


app.mount("/audio", StaticFiles(directory=config.AUDIO_DIR), name="audio")
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
