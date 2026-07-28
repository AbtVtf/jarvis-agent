"""Audio + transcript -> viseme timeline via forced alignment.

Pipeline: MMS_FA (torchaudio) aligns transcript characters to audio frames,
giving word time spans; g2p_en converts each word to ARPAbet phonemes; phoneme
durations are distributed inside the word span (vowels weighted longer); each
phoneme maps to one of the 15 Oculus visemes.

Timeline format: list of {"v": viseme, "t0": sec, "t1": sec, "w": weight}.
"""

import re

import torch
import torchaudio

# ARPAbet -> (viseme, weight). Diphthongs collapse to their dominant shape.
PHONE_TO_VISEME = {
    "AA": ("aa", 1.0), "AO": ("aa", 1.0), "AH": ("aa", 0.7), "AY": ("aa", 1.0),
    "AW": ("aa", 1.0), "HH": ("aa", 0.25),
    "AE": ("E", 1.0), "EH": ("E", 1.0), "EY": ("E", 1.0),
    "IH": ("ih", 1.0), "IY": ("ih", 1.0), "Y": ("ih", 0.8),
    "OW": ("oh", 1.0), "OY": ("oh", 1.0),
    "UW": ("ou", 1.0), "UH": ("ou", 0.9), "W": ("ou", 0.9),
    "M": ("PP", 1.0), "B": ("PP", 1.0), "P": ("PP", 1.0),
    "F": ("FF", 1.0), "V": ("FF", 1.0),
    "TH": ("TH", 1.0), "DH": ("TH", 0.8),
    "T": ("DD", 1.0), "D": ("DD", 1.0),
    "K": ("kk", 1.0), "G": ("kk", 1.0), "NG": ("kk", 0.8),
    "CH": ("CH", 1.0), "JH": ("CH", 1.0), "SH": ("CH", 1.0), "ZH": ("CH", 1.0),
    "S": ("SS", 1.0), "Z": ("SS", 1.0),
    "N": ("nn", 1.0), "L": ("nn", 1.0),
    "R": ("RR", 1.0), "ER": ("RR", 1.0),
}
VOWELS = {"AA", "AO", "AH", "AY", "AW", "AE", "EH", "EY", "IH", "IY",
          "OW", "OY", "UW", "UH", "ER"}


class LipSync:

  def __init__(self, device: str = "cuda"):
    from g2p_en import G2p  # noqa: PLC0415 (heavy import, keep lazy)

    self.device = device
    self.bundle = torchaudio.pipelines.MMS_FA
    self.model = self.bundle.get_model(with_star=False).to(device).eval()
    self.dictionary = self.bundle.get_dict(star=None)
    self.g2p = G2p()

  def timeline(self, wav_path: str, transcript: str):
    words = re.findall(r"[a-z']+", transcript.lower())
    if not words:
      return []

    waveform, sr = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    target_sr = self.bundle.sample_rate
    if sr != target_sr:
      waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    with torch.inference_mode():
      emission, _ = self.model(waveform.to(self.device))

    # Tokenize; drop characters MMS_FA doesn't know (e.g. digits already gone).
    tokenized = [[self.dictionary[c] for c in w if c in self.dictionary]
                 for w in words]
    kept = [(w, t) for w, t in zip(words, tokenized) if t]
    if not kept:
      return []
    words = [w for w, _ in kept]
    targets = torch.tensor(
        [tok for _, toks in kept for tok in toks],
        dtype=torch.int32, device=self.device,
    ).unsqueeze(0)

    aligned, scores = torchaudio.functional.forced_align(
        emission, targets, blank=0)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0])

    frame_dt = waveform.shape[1] / target_sr / emission.shape[1]

    # Group char spans back into words.
    word_spans = []
    i = 0
    for _, toks in kept:
      chars = spans[i:i + len(toks)]
      i += len(toks)
      word_spans.append((chars[0].start * frame_dt, (chars[-1].end + 1) * frame_dt))

    segments = []
    for word, (t0, t1) in zip(words, word_spans):
      phones = [re.sub(r"\d", "", p) for p in self.g2p(word)]
      phones = [p for p in phones if p in PHONE_TO_VISEME]
      if not phones:
        continue
      weights = [1.6 if p in VOWELS else 1.0 for p in phones]
      total = sum(weights)
      t = t0
      for phone, dur_w in zip(phones, weights):
        dt = (t1 - t0) * dur_w / total
        vis, w = PHONE_TO_VISEME[phone]
        segments.append({"v": vis, "t0": round(t, 4),
                         "t1": round(t + dt, 4), "w": w})
        t += dt

    # Merge consecutive identical visemes.
    merged = []
    for seg in segments:
      if merged and merged[-1]["v"] == seg["v"] and \
         abs(merged[-1]["t1"] - seg["t0"]) < 0.02:
        merged[-1]["t1"] = seg["t1"]
        merged[-1]["w"] = max(merged[-1]["w"], seg["w"])
      else:
        merged.append(seg)
    return merged
