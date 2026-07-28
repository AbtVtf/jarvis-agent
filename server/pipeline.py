"""Streaming speech pipeline: LLM text deltas -> sentence chunks -> TTS +
lip-sync -> websocket messages.

The face starts talking after the FIRST sentence is synthesized, while the
model is still generating the rest.
"""

import asyncio
import os
import re
import uuid

from server import config
from server.tts import clean_for_tts

_BOUNDARY = re.compile(r"(?<=[.!?…])[\s\"')\]]*\s+")
FIRST_FLUSH_MIN = 12   # chars — flush the first sentence ASAP
LATER_FLUSH_MIN = 50   # chars — batch later sentences a bit for TTS quality


class VoicePipeline:

  def __init__(self, tts, lipsync):
    self.tts = tts
    self.lipsync = lipsync
    self._gpu_lock = asyncio.Lock()

  async def _synth_chunk(self, text: str, idx: int) -> dict | None:
    text_clean = clean_for_tts(text)
    if not text_clean or not re.search(r"[a-zA-Z0-9]", text_clean):
      return None
    clip_id = f"{uuid.uuid4().hex[:10]}_{idx}"
    wav_path = os.path.join(config.AUDIO_DIR, f"{clip_id}.wav")
    loop = asyncio.get_running_loop()
    async with self._gpu_lock:
      await loop.run_in_executor(None, self.tts.synthesize, text_clean, wav_path)
      timeline = await loop.run_in_executor(
          None, self.lipsync.timeline, wav_path, text_clean)
    return {"type": "chunk", "idx": idx, "text": text,
            "audio": f"/audio/{clip_id}.wav", "timeline": timeline}

  async def speak_deltas(self, delta_gen, send):
    """Consume ('text', delta) events; stream synthesized chunks via send().

    Chunks are synthesized strictly in order but concurrently with ongoing
    LLM generation. Returns the full reply text.
    """
    buffer = ""
    full = ""
    idx = 0
    synth_tasks: list[asyncio.Task] = []

    async def flush(text):
      nonlocal idx
      task = asyncio.create_task(self._synth_chunk(text, idx))
      idx += 1
      synth_tasks.append(task)

    async for kind, delta in delta_gen:
      if kind != "text":
        continue
      full += delta
      buffer += delta
      await send({"type": "caption_delta", "text": delta})
      # Flush complete sentences out of the buffer.
      while True:
        m = _BOUNDARY.search(buffer)
        if not m:
          break
        sentence, rest = buffer[:m.start() + 1], buffer[m.end():]
        min_len = FIRST_FLUSH_MIN if idx == 0 else LATER_FLUSH_MIN
        if len(sentence.strip()) < min_len and rest:
          # Too short on its own — merge with what follows.
          break
        await flush(sentence.strip())
        buffer = rest
      # Ship any chunks that are already done, in order.
      while synth_tasks and synth_tasks[0].done():
        chunk = synth_tasks.pop(0).result()
        if chunk:
          await send(chunk)

    if buffer.strip():
      await flush(buffer.strip())
    for task in synth_tasks:
      chunk = await task
      if chunk:
        await send(chunk)
    await send({"type": "chunks_done"})
    return full.strip()

  async def speak_text(self, text: str, send):
    """Speak a fixed string (announcements) through the same chunk protocol."""

    async def gen():
      yield ("text", text)

    return await self.speak_deltas(gen(), send)
