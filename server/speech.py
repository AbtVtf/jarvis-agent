"""Speech input: wake word ("hey jarvis"), VAD utterance capture, Whisper STT.

The widget streams 16 kHz mono s16le PCM over a websocket. Each connection
owns a MicSession with two modes:
  wake    - passive: run openWakeWord on 80 ms chunks; on detection -> "wake"
  capture - active: collect an utterance, end it via silero-VAD silence,
            transcribe with faster-whisper -> ("utterance", text)
"""

import time

import numpy as np
import torch

from server import config

SR = 16000
OWW_CHUNK = 1280   # 80 ms — openWakeWord's expected hop
VAD_CHUNK = 512    # 32 ms — silero's expected hop at 16 kHz
SILENCE_END_S = 0.8
NO_SPEECH_TIMEOUT_S = 7.0
MAX_UTTERANCE_S = 30.0


class SpeechEngine:
  """Heavy models, shared across sessions."""

  def __init__(self):
    import os
    from faster_whisper import WhisperModel  # noqa: PLC0415 (slow imports)
    import openwakeword
    from openwakeword.model import Model as OWWModel
    from silero_vad import load_silero_vad

    self._oww_cls = OWWModel
    self._oww_path = os.path.join(
        os.path.dirname(openwakeword.__file__), "resources", "models",
        f"{config.WAKE_WORD}_v0.1.onnx")
    self.whisper = WhisperModel(config.STT_MODEL, device="cuda",
                                compute_type="float16")
    self.vad = load_silero_vad()

  def new_wake_model(self):
    return self._oww_cls(wakeword_model_paths=[self._oww_path])

  def transcribe(self, audio_f32: np.ndarray) -> str:
    segments, _ = self.whisper.transcribe(
        audio_f32, beam_size=1, language="en", vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()

  def vad_prob(self, chunk_f32: np.ndarray) -> float:
    return float(self.vad(torch.from_numpy(chunk_f32), SR).item())

  def reset_vad(self):
    self.vad.reset_states()


class MicSession:

  def __init__(self, engine: SpeechEngine):
    self.engine = engine
    self.oww = engine.new_wake_model()
    self.mode = "wake"
    self._buf = np.zeros(0, dtype=np.int16)
    self._utt = []
    self._speech_seen = False
    self._last_speech_t = 0.0
    self._capture_started_t = 0.0
    self._wake_mute_until = 0.0

  def set_mode(self, mode: str):
    self.mode = mode
    self._buf = np.zeros(0, dtype=np.int16)
    if mode == "capture":
      self._utt = []
      self._speech_seen = False
      self._capture_started_t = time.time()
      self.engine.reset_vad()
    if mode == "wake":
      self.oww.reset()
      # The wake model's internal window can still hold the trigger phrase;
      # mute briefly so one "hey jarvis" can't fire twice.
      self._wake_mute_until = time.time() + 1.5

  def feed(self, pcm: bytes) -> list[tuple]:
    """Feed raw PCM; returns a list of events."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    self._buf = np.concatenate([self._buf, samples])
    if self.mode == "wake":
      return self._feed_wake()
    if self.mode == "capture":
      return self._feed_capture()
    return []

  def _feed_wake(self) -> list[tuple]:
    events = []
    while len(self._buf) >= OWW_CHUNK:
      chunk, self._buf = self._buf[:OWW_CHUNK], self._buf[OWW_CHUNK:]
      scores = self.oww.predict(chunk)
      if time.time() < self._wake_mute_until:
        continue
      if max(scores.values(), default=0.0) >= config.WAKE_THRESHOLD:
        events.append(("wake",))
        self.set_mode("capture")
        break
    return events

  def finish(self) -> list[tuple]:
    """Force-end a capture now (push-to-talk release)."""
    if self.mode != "capture":
      return []
    audio = np.concatenate(self._utt) if self._utt else np.zeros(1, np.float32)
    self.set_mode("wake")
    if len(audio) < SR // 4:
      return [("timeout",)]
    text = self.engine.transcribe(audio)
    return [("utterance", text)] if text else [("timeout",)]

  def _feed_capture(self) -> list[tuple]:
    events = []
    now = time.time()
    while len(self._buf) >= VAD_CHUNK:
      chunk, self._buf = self._buf[:VAD_CHUNK], self._buf[VAD_CHUNK:]
      f32 = chunk.astype(np.float32) / 32768.0
      self._utt.append(f32)
      prob = self.engine.vad_prob(f32)
      if prob >= 0.5:
        self._speech_seen = True
        self._last_speech_t = now

    utt_len_s = sum(len(c) for c in self._utt) / SR
    if not self._speech_seen:
      if now - self._capture_started_t > NO_SPEECH_TIMEOUT_S:
        events.append(("timeout",))
        self.set_mode("wake")
    elif (now - self._last_speech_t > SILENCE_END_S
          or utt_len_s > MAX_UTTERANCE_S):
      audio = np.concatenate(self._utt) if self._utt else np.zeros(1, np.float32)
      self.set_mode("wake")
      text = self.engine.transcribe(audio)
      if text:
        events.append(("utterance", text))
      else:
        events.append(("timeout",))
    return events
