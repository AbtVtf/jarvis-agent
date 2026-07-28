"""Chatterbox TTS wrapper.

Voice customization: drop a short (~10 s) clean speech recording at
data/voice/reference.wav and Jarvis speaks in that voice (Chatterbox voice
cloning) — only use a voice you have the right to clone. Delivery is tuned
via JARVIS_VOICE_EXAGGERATION (emotion intensity, default 0.35) and
JARVIS_VOICE_CFG (pacing/adherence, default 0.4).
"""

import math
import os
import re

import torch
import torchaudio

from server import config

VOICE_WAV = os.environ.get(
    "JARVIS_VOICE_WAV", os.path.join(config.ROOT, "data", "voice",
                                     "reference.wav"))
EXAGGERATION = float(os.environ.get("JARVIS_VOICE_EXAGGERATION", "0.28"))
CFG_WEIGHT = float(os.environ.get("JARVIS_VOICE_CFG", "0.4"))
# "digital" layers a subtle vocoder-ish treatment over the natural voice.
VOICE_FX = os.environ.get("JARVIS_VOICE_FX", "digital")
FX_DEPTH = float(os.environ.get("JARVIS_VOICE_FX_DEPTH", "0.4"))


def robotize(wav: torch.Tensor, sr: int, depth: float) -> torch.Tensor:
  """Subtle digital-being treatment: AM shimmer, detuned double, thin lows."""
  n = wav.shape[-1]
  peak = wav.abs().max().clamp(min=1e-6)

  # Amplitude shimmer at 30 Hz — the classic "synthetic larynx" cue.
  t = torch.arange(n, dtype=torch.float32) / sr
  am = 1.0 - 0.22 * depth + 0.22 * depth * torch.sin(2 * math.pi * 30.0 * t)
  out = wav * am

  # Detuned double a few cents down, mixed underneath (vocoder feel).
  detuned = torchaudio.functional.resample(wav, sr, int(sr * 1.008))
  detuned = detuned[..., :n]
  if detuned.shape[-1] < n:
    detuned = torch.nn.functional.pad(detuned, (0, n - detuned.shape[-1]))
  out = (1.0 - 0.28 * depth) * out + 0.28 * depth * detuned

  # Thin out the body a little so it reads as a speaker, not a chest.
  out = torchaudio.functional.highpass_biquad(out, sr, 110.0)
  return out * (peak / out.abs().max().clamp(min=1e-6))


def clean_for_tts(text: str) -> str:
  text = re.sub(r"[*_`#>\[\]()~]", "", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text


class TTS:

  def __init__(self, device: str = "cuda"):
    from chatterbox.tts import ChatterboxTTS  # noqa: PLC0415 (slow import)

    self.model = ChatterboxTTS.from_pretrained(device=device)
    self.voice_wav = VOICE_WAV if os.path.isfile(VOICE_WAV) else None
    if self.voice_wav:
      print(f"[tts] using voice reference: {self.voice_wav}")

  def synthesize(self, text: str, out_path: str) -> str:
    kwargs = {"exaggeration": EXAGGERATION, "cfg_weight": CFG_WEIGHT}
    if self.voice_wav:
      kwargs["audio_prompt_path"] = self.voice_wav
    wav = self.model.generate(clean_for_tts(text), **kwargs).cpu()
    if VOICE_FX == "digital" and FX_DEPTH > 0:
      wav = robotize(wav, self.model.sr, FX_DEPTH)
    torchaudio.save(out_path, wav, self.model.sr)
    return out_path
