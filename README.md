# Jarvis

A fully local, voice-first orchestration layer over your Claude Code agents,
with a 3D talking face (Google GNM head + Chatterbox TTS + real lip sync).

You talk to Jarvis. Jarvis spawns headless Claude Code agents in your project
folders, watches what they do in real time, relays your follow-ups, and
speaks up when something finishes. Jarvis never writes code himself — his
brain is a local LLM whose only powers are orchestration tools.

## Quick start

```bash
./jarvis          # starts ollama + server + desktop widget
./jarvis status
./jarvis stop
```

Then either:
- **say "hey jarvis"** (widget mic is always listening once enabled),
- **click the mic** for push-to-talk (click again to send),
- press **Ctrl+Shift+J** to toggle the widget,
- or open http://127.0.0.1:8710 in a browser.

Jarvis is also in your app launcher ("Jarvis" in GNOME Activities — pin it
to the dash for one-click start after a full stop).

First server start takes ~60–90 s (loads Gemma 4, Chatterbox, Whisper, the
aligner and the wake-word model — ~17 GB VRAM total on the 3090).

## What Jarvis can do

- "Spawn an agent in ardy to fix the failing tests" → picks the folder,
  writes a proper task brief, starts `claude -p` there.
- "What's everyone doing?" → live status from the stream-json event feeds.
- "Tell the parletto agent to also update the docs" → resumes that agent's
  session with context intact.
- Speaks up on completion, plus a desktop notification from the widget.
- "Remember that I prefer X" → stored in SQLite, in context every session.

## Daily-life abilities (no tokens needed)

- **Reminders/timers** — "remind me in 20 minutes to check the oven" →
  spoken announcement + notification when due; survives restarts.
- **Todos** — "add X to my list", "what's on my list", "mark X done".
- **Notes** — "note that …", "what did I note about …".
- **Weather** — Open-Meteo, keyless; "set my location to X" persists.
- **Briefing** — "give me my briefing": weather, reminders, todos, agents,
  dirty repos, top Hacker News. Schedule it daily: "brief me every morning
  at 8:30".
- **Repo sweep** — "what's the state of my repos" (uncommitted changes).
- **System health** — CPU/RAM/disk/GPU on demand; proactive spoken alerts
  for low disk or hot GPU (throttled to once per 2 h).
- **Tech news** — Hacker News front page, keyless.
- **Media & volume** — "pause the music", "next", "volume up", "mute"
  (MPRIS via gdbus + wpctl).
- **Clipboard** — "copy this to my clipboard: …", "read my clipboard".
- **Schedules** — recurring daily/weekday automations: spoken briefings,
  fixed announcements, or scheduled agent runs.

Wants tokens later: Google Calendar/Gmail, Spotify search, home automation.

## Architecture

```
widget (Electron, frameless, tray, mic)      browser fallback
        └── ws://127.0.0.1:8710 ────────────────┘
server (FastAPI):
  /ws/audio  16k PCM → openWakeWord("hey jarvis") → silero-VAD → faster-whisper
  /ws/ui     state / caption deltas / audio+viseme chunks / agent status / notify
  brain      ollama (gemma4:12b) streaming + tool calls
  orchestrator  claude -p --output-format stream-json per project; resume via
                session ids; permission_denials surfaced
  pipeline   sentence-chunked Chatterbox TTS + MMS_FA forced alignment →
             face starts speaking after the FIRST sentence
  memory     SQLite: messages, facts, agent runs, rolling summary
```

The face: GNM head exported with 15 Oculus visemes + blink as morph targets
(authored via landmark-constrained least squares in GNM's expression space —
see tools/build_visemes.py), rendered in three.js.

## Integrated systems (jarvis.md)

Any project under your projects root becomes a Jarvis-aware **system** by
adding a `jarvis.md` at its root: frontmatter with `name`, `description`,
`data:` globs (files Jarvis may read) and `commands:` (shell commands Jarvis
may run, cwd'd to the project), followed by free-form usage notes Jarvis
follows verbatim. Jarvis can only read the declared files and run the
declared commands — the manifest is the contract.

First integration: **meal-planner** (`~/Documents/GitHub/meal-planner/jarvis.md`)
— ask "what's for dinner this week?" or "walk me through the carbonara" for
one-step-at-a-time kitchen mode; `compare_prices` runs the live store
comparison. When Jarvis spawns agents to build new tools, he asks them to
write a jarvis.md so the result plugs itself in.

## Brain backends

Default: **OpenRouter** (`deepseek/deepseek-v4-flash`, ~$0.0005/turn) with
the key read from `data/openrouter.key` (chmod 600, gitignored). If
OpenRouter is unreachable mid-conversation, the turn silently retries on
local **ollama** (gemma4:12b), so Jarvis works offline. Force local with
`JARVIS_BRAIN=ollama`. With the cloud brain active, gemma stays unloaded and
GPU use drops to ~8 GB.

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `JARVIS_MODEL` | `gemma4:12b` | brain model (qwen3.5:9b also pulled) |
| `JARVIS_PROJECTS_ROOT` | `~/Documents/GitHub` | where agents may work |
| `JARVIS_AGENT_PERMISSIONS` | `auto` | claude permission mode for agents |
| `JARVIS_MAX_AGENTS` | 6 | concurrent agent cap |
| `JARVIS_STT_MODEL` | `distil-large-v3` | faster-whisper model |
| `JARVIS_WAKE_THRESHOLD` | 0.5 | wake-word sensitivity |
| `JARVIS_FACE_THEME` | `digital` | `digital` (blue hologram) or `human`; re-run `tools/export_head.py` after changing |
| `JARVIS_OPENROUTER_MODEL` | `deepseek/deepseek-v4-flash` | cloud brain model |
| `JARVIS_BRAIN` | auto | `openrouter` or `ollama` |
| `JARVIS_VOICE_EXAGGERATION` | 0.28 | TTS emotion intensity (0–1) |
| `JARVIS_VOICE_CFG` | 0.4 | TTS pacing/adherence |
| `JARVIS_VOICE_FX` | `digital` | robotic post-processing (`none` to disable) |
| `JARVIS_VOICE_FX_DEPTH` | 0.4 | how robotic (0–1) |

**Custom voice**: drop a clean ~10 s speech recording at
`data/voice/reference.wav` and restart — Chatterbox clones that voice for
Jarvis. Only use a voice you have the right to use.

Agent permissions: `auto` lets agents edit files and run safe in-cwd
commands; blocked actions are reported back honestly. Saying "full autonomy"
to Jarvis spawns that agent with `--dangerously-skip-permissions` — use
deliberately.

## Notes & troubleshooting

- **Electron sandbox**: this kernel blocks unprivileged user namespaces, so
  the launcher uses `--no-sandbox` (localhost-only content). To restore the
  sandbox: `sudo chown root:root widget/node_modules/electron/dist/chrome-sandbox
  && sudo chmod 4755 widget/node_modules/electron/dist/chrome-sandbox`, then
  remove the flag from `./jarvis`.
- **Tray on GNOME** needs the AppIndicator extension; without it you still
  have Ctrl+Shift+J and wake word.
- Whisper OOM at startup usually means ollama has two models resident:
  `build/ollama/bin/ollama ps`, unload the extra one.
- The GNM head model lives in `~/Documents/GitHub/ardy/GNM` (installed
  editable into `.venv`).
