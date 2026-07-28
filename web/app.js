import * as THREE from "./vendor/three.module.min.js";

const canvas = document.getElementById("scene");
const statusEl = document.getElementById("status");
const captionEl = document.getElementById("caption");
const msgEl = document.getElementById("msg");
const sendEl = document.getElementById("send");
const micEl = document.getElementById("mic");
const agentsEl = document.getElementById("agents");
const IS_WIDGET = new URLSearchParams(location.search).has("widget") ||
  (window.jarvisShell && window.jarvisShell.isWidget);

// ---------- head loading (same packed format as v1) ----------

async function loadHead() {
  const [manifest, buf] = await Promise.all([
    fetch("assets/head.json").then((r) => r.json()),
    fetch("assets/head.bin").then((r) => r.arrayBuffer()),
  ]);
  const block = (name) => {
    const b = manifest.blocks[name];
    const n = b.shape.reduce((a, x) => a * x, 1);
    if (b.dtype === "float32") return new Float32Array(buf, b.offset, n);
    if (b.dtype === "uint32") return new Uint32Array(buf, b.offset, n);
    if (b.dtype === "uint8") return new Uint8Array(buf, b.offset, n);
    throw new Error("dtype " + b.dtype);
  };
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(block("positions"), 3));
  geo.setAttribute("color", new THREE.BufferAttribute(block("colors"), 3, true));
  geo.setIndex(new THREE.BufferAttribute(block("triangles"), 1));
  geo.morphTargetsRelative = true;
  geo.morphAttributes.position = manifest.morphNames.map(
    (n) => new THREE.BufferAttribute(block("morph_" + n), 3));
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.38, metalness: 0.15,
    emissive: 0x0a1e3c, emissiveIntensity: 0.55 }));
  mesh.morphTargetInfluences = new Array(manifest.morphNames.length).fill(0);
  return { mesh,
    morphIndex: Object.fromEntries(manifest.morphNames.map((n, i) => [n, i])) };
}

// ---------- scene ----------

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);
const camera = new THREE.PerspectiveCamera(23, 1, 0.05, 10);
camera.position.set(0, 0.26, 0.78);
camera.lookAt(0, 0.24, 0);
scene.add(new THREE.HemisphereLight(0xbfe0ff, 0x0a1428, 0.9));
const key = new THREE.DirectionalLight(0xcfe6ff, 1.9);
key.position.set(0.5, 0.7, 0.8);
scene.add(key);
const rim = new THREE.DirectionalLight(0x3fd0ff, 2.2);
rim.position.set(-0.7, 0.3, -0.5);
scene.add(rim);
const rim2 = new THREE.DirectionalLight(0x2f7fff, 1.4);
rim2.position.set(0.7, 0.2, -0.6);
scene.add(rim2);

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

// ---------- animation ----------

let head = null, morphIndex = {}, targets = null;
const group = new THREE.Group();
scene.add(group);
let nextBlinkAt = performance.now() / 1000 + 2.5;
let blinkPhase = -1;
let stateName = "loading";
let gazeTwitch = 0;
const ATTACK = 0.045, RELEASE = 0.075, LOOKAHEAD = 0.04, BLINK_LEN = 0.18;

// Playback queue of {audio: HTMLAudio, timeline, text}
const queue = [];
let current = null;
let chunksDone = true;

function activeViseme() {
  if (!current) return null;
  const t = current.audio.currentTime + LOOKAHEAD;
  for (const seg of current.timeline) {
    if (t >= seg.t0 - 0.012 && t < seg.t1) return seg;
  }
  return null;
}

function pump() {
  if (current || !queue.length) return;
  current = queue.shift();
  current.audio.play().catch((e) => { console.error("play failed", e); skip(); });
  current.audio.onended = skip;
  setStatus("speaking");
}
// Playback must advance even when the window is hidden (rAF pauses there).
setInterval(pump, 120);

function skip() {
  current = null;
  if (queue.length) pump();
  else if (chunksDone) setStatus("idle");
}

function stopPlayback() {
  queue.length = 0;
  chunksDone = true;
  if (current) { current.audio.onended = null; current.audio.pause(); current = null; }
  setStatus("idle");
}

let lastT = performance.now() / 1000;
function tick() {
  requestAnimationFrame(tick);
  const now = performance.now() / 1000;
  const dt = Math.min(now - lastT, 0.1);
  lastT = now;
  if (!head) return;
  targets.fill(0);
  const seg = activeViseme();
  if (seg && seg.v !== "sil") targets[morphIndex[seg.v]] = seg.w;

  // "Accessing the mainframe": while thinking, eyes roll up and the pupils
  // twitch and pulse until the reply starts.
  if (stateName === "thinking" && !current) {
    targets[morphIndex.eyes_up] = 0.85;
    gazeTwitch = (gazeTwitch + (Math.random() - 0.5) * dt * 12) * 0.93;
    gazeTwitch = Math.max(-0.35, Math.min(0.35, gazeTwitch));
    targets[morphIndex.eyes_side] = gazeTwitch;
    targets[morphIndex.pupil_wide] =
      0.5 + 0.3 * Math.sin(now * 3.1) + 0.12 * Math.sin(now * 7.7);
  }

  if (blinkPhase >= 0) {
    blinkPhase += dt;
    const p = blinkPhase / BLINK_LEN;
    if (p >= 1) { blinkPhase = -1; nextBlinkAt = now + 1.8 + Math.random() * 4.2; }
    else targets[morphIndex.blink] = Math.sin(Math.PI * Math.min(p, 1));
  } else if (now >= nextBlinkAt) blinkPhase = 0;

  const inf = head.mesh.morphTargetInfluences;
  for (let i = 0; i < inf.length; i++) {
    const tau = targets[i] > inf[i] ? ATTACK : RELEASE;
    inf[i] += (targets[i] - inf[i]) * (1 - Math.exp(-dt / tau));
  }
  const sway = current ? 1.6 : 1.0;
  group.rotation.y = Math.sin(now * 0.31) * 0.035 * sway;
  group.rotation.x = Math.sin(now * 0.23 + 1.7) * 0.018 * sway;
  group.rotation.z = Math.sin(now * 0.17 + 0.6) * 0.008;
  renderer.render(scene, camera);
}

// ---------- websocket ----------

let ws = null;
let captionText = "";
let discardChunks = false;

function setStatus(s) {
  stateName = s;
  statusEl.textContent = s;
  statusEl.className = s;
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws/ui`);
  ws.onopen = () => { setStatus("idle"); msgEl.disabled = sendEl.disabled = false; };
  ws.onclose = () => { setStatus("reconnecting…"); setTimeout(connect, 2000); };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
}

function handle(msg) {
  switch (msg.type) {
    case "state":
      if (msg.state === "idle" && (current || queue.length)) break;
      setStatus(msg.state);
      if (msg.state === "listening" && window.jarvisShell) window.jarvisShell.show();
      if (msg.state === "listening") micEl.classList.add("hot");
      else { micEl.classList.remove("hot"); capturing = false; }
      break;
    case "user_text":
      discardChunks = false;
      stopPlayback();
      captionText = "";
      captionEl.innerHTML = `<span class="you"></span>`;
      captionEl.querySelector(".you").textContent = "you: " + msg.text;
      break;
    case "caption_delta":
      captionText += msg.text;
      captionEl.textContent = captionText;
      break;
    case "chunk": {
      if (discardChunks) break;
      const audio = new Audio(msg.audio);
      queue.push({ audio, timeline: msg.timeline, text: msg.text });
      chunksDone = false;
      break;
    }
    case "chunks_done":
      chunksDone = true;
      break;
    case "wake":
      if (window.jarvisShell) window.jarvisShell.show();
      stopPlayback();
      break;
    case "notify":
      if (window.jarvisShell) window.jarvisShell.notify(msg.title, msg.body);
      break;
    case "agents":
      renderAgents(msg.agents || []);
      break;
    case "error":
      captionEl.textContent = "error: " + msg.text;
      break;
  }
}

// Finished agent cards linger for a bit, then fade out (click dismisses).
const CARD_TTL_S = 90;
let lastAgents = [];
const dismissed = new Set();

function renderAgents(agents) {
  if (agents) {
    const now = Date.now() / 1000;
    lastAgents = agents.map((a) => ({
      ...a, hideAt: a.status === "running" || a.status === "starting"
        ? Infinity : now + CARD_TTL_S - (a.finished_ago_s || 0) }));
  }
  const now = Date.now() / 1000;
  agentsEl.innerHTML = "";
  for (const a of lastAgents.slice(-8)) {
    if (dismissed.has(a.id) || now > a.hideAt) continue;
    const div = document.createElement("div");
    div.className = "agent";
    const mins = Math.floor(a.elapsed_s / 60);
    const dur = mins ? `${mins}m` : `${a.elapsed_s}s`;
    div.innerHTML = `<span class="dot ${a.status}"></span><b></b> · ${dur}
      <span class="act"></span>`;
    div.querySelector("b").textContent = a.project;
    div.querySelector(".act").textContent =
      a.status === "running" ? a.last_activity : a.status;
    div.title = a.task + `  [${a.id}] — click to dismiss`;
    div.addEventListener("click", () => { dismissed.add(a.id); renderAgents(); });
    agentsEl.appendChild(div);
  }
}
setInterval(() => renderAgents(), 5000);

// ---------- input ----------

function sendText() {
  const text = msgEl.value.trim();
  if (!text || !ws || ws.readyState !== 1) return;
  msgEl.value = "";
  stopPlayback();
  ws.send(JSON.stringify({ type: "text", text }));
}
sendEl.addEventListener("click", sendText);
msgEl.addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });
addEventListener("keydown", (e) => {
  if (e.key === "Escape") { stopPlayback(); ws?.send(JSON.stringify({ type: "interrupt" })); }
});

// ---------- microphone ----------

let micStream = null, micCtx = null, micWs = null;
let capturing = false;

async function micOn() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true } });
  } catch (e) { console.error("mic denied", e); return false; }
  micWs = new WebSocket(`ws://${location.host}/ws/audio`);
  micWs.binaryType = "arraybuffer";
  micCtx = new AudioContext();
  await micCtx.audioWorklet.addModule("mic-worklet.js");
  const src = micCtx.createMediaStreamSource(micStream);
  const node = new AudioWorkletNode(micCtx, "mic-downsampler");
  node.port.onmessage = (e) => {
    if (micWs.readyState === 1) micWs.send(e.data);
  };
  src.connect(node);
  micEl.classList.add("on");
  msgEl.placeholder = "Press 🎤 and talk, say “hey jarvis”, or type…";
  return true;
}

function micOff() {
  micStream?.getTracks().forEach((t) => t.stop());
  micCtx?.close();
  micWs?.close();
  micStream = micCtx = micWs = null;
  capturing = false;
  micEl.classList.remove("on", "hot");
  msgEl.placeholder = "Talk or type…";
}

function micCmd(cmd) {
  if (micWs?.readyState === 1) micWs.send(JSON.stringify({ cmd }));
}

// Click = push-to-talk: press, speak, press again to send.
// The wake word works any time the mic is on. Double-click = mic fully off.
let clickTimer = null;
micEl.addEventListener("click", async () => {
  if (clickTimer) { // double-click → mic off
    clearTimeout(clickTimer);
    clickTimer = null;
    micOff();
    return;
  }
  clickTimer = setTimeout(async () => {
    clickTimer = null;
    if (!micStream) {
      if (!(await micOn())) return;
      stopPlayback();
      discardChunks = true;
      micCmd("capture");
      capturing = true;
    } else if (!capturing) {
      stopPlayback();
      discardChunks = true;
      micCmd("capture");
      capturing = true;
    } else {
      micCmd("finish");
      capturing = false;
    }
  }, 260);
});

// ---------- boot ----------

// Debug handle (used by automated tests; harmless in production).
window.__jarvis = { setState: (s) => setStatus(s) };

if (IS_WIDGET) {
  document.body.classList.add("widget");
  document.getElementById("winmin").addEventListener("click",
    () => window.jarvisShell?.minimize());
  document.getElementById("winhide").addEventListener("click",
    () => window.jarvisShell?.hide());
}

loadHead().then((h) => {
  head = h;
  morphIndex = h.morphIndex;
  targets = new Float32Array(h.mesh.morphTargetInfluences.length);
  h.mesh.position.set(0, -0.235, 0);
  const pivot = new THREE.Group();
  pivot.position.set(0, 0.235, 0);
  pivot.add(h.mesh);
  group.add(pivot);
  tick();
  connect();
  if (IS_WIDGET) micOn();  // Electron pre-grants mic permission.
});
