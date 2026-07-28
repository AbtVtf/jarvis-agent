// Downsamples mic input to 16 kHz s16le and posts 80 ms frames.
class MicDownsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.readPos = 0;
    this.input = [];
    this.out = new Int16Array(1280);
    this.outPos = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    this.input.push(Float32Array.from(ch));
    // Flatten lazily: consume with linear interpolation.
    let flat = this.input.length === 1 ? this.input[0] : concat(this.input);
    this.input = [flat];
    while (this.readPos + this.ratio < flat.length) {
      const i = Math.floor(this.readPos);
      const frac = this.readPos - i;
      const s = flat[i] * (1 - frac) + flat[i + 1] * frac;
      this.out[this.outPos++] = Math.max(-32768, Math.min(32767, s * 32768));
      this.readPos += this.ratio;
      if (this.outPos === this.out.length) {
        this.port.postMessage(this.out.buffer.slice(0));
        this.outPos = 0;
      }
    }
    const consumed = Math.floor(this.readPos);
    this.input = [flat.subarray(consumed)];
    this.readPos -= consumed;
    return true;
  }
}

function concat(chunks) {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

registerProcessor("mic-downsampler", MicDownsampler);
