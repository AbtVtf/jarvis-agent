# Jarvis

A fully local, voice-first companion over your Claude Code sessions, with a
3D talking face (Google GNM head + Chatterbox TTS + real lip sync).

You work in Claude Code sessions wherever you like — terminals or the
CloudCLI web UI. Whenever ANY session finishes a turn, Jarvis is notified
automatically (a global Claude Code Stop hook), reads the message straight
from the session transcript, and speaks his own summary of it. You answer
by voice; Jarvis delivers your words into the right session through
CloudCLI, where the conversation continues visibly. Claude needs no MCP, no
helper scripts, no cooperation — it just receives prompts.

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

First server start takes ~60–90 s (loads Gemma 4, Chatterbox, Whisper, the
aligner and the wake-word model — ~17 GB VRAM total on the 3090).

## How the Claude Code integration works

```
any Claude Code session finishes a turn
  └─ Stop hook (~/.claude/settings.json): curl → POST /claude-event
       └─ Jarvis reads the last assistant message from the transcript
          (~/.claude/projects/<project>/<session>.jsonl)
            └─ brain composes 1–2 spoken sentences → face speaks them
you answer by voice
  └─ brain tool send_to_session → CloudCLI ws chat.send → the session
     runs your words as its next prompt → its result is announced the
     same way when it finishes
```

- **Announce**: every turn of every session, with project attribution.
  Tool-only turns with no visible text are skipped.
- **Reply routing**: "tell them yes" targets the most recently announced
  session; name a project ("tell the padel session…") to target another;
  Jarvis asks when it's ambiguous.
- **Reading**: "what did the parletto session say?" → read_session quotes
  the transcript.
- **Overview**: "what's going on?" → sessions_overview lists recently
  active sessions across all projects.

Requires [CloudCLI](https://github.com/siteboon/claudecodeui) running
locally for the reply path (announcements work without it). Credentials in
`data/cloudcli.json` (gitignored): `{"url", "username", "password"}` —
update it if you change your CloudCLI password.

## Brain tools

`sessions_overview`, `read_session`, `send_to_session`, `remember`
(durable facts in SQLite, in context every session). That's the whole
surface — Jarvis never writes code and never spawns anything.

## Architecture

```
widget (Electron, frameless, tray, mic)      browser fallback
        └── ws://127.0.0.1:8710 ────────────────┘
server (FastAPI):
  /ws/audio      16k PCM → openWakeWord("hey jarvis") → silero-VAD → faster-whisper
  /ws/ui         state / caption deltas / audio+viseme chunks / session cards / notify
  /claude-event  Stop-hook intake: transcript read → announcement
  brain          OpenRouter or ollama streaming + tool calls
  sessions       transcript reader + CloudCLI client (login, ws chat.send)
  pipeline       sentence-chunked Chatterbox TTS + MMS_FA forced alignment →
                 face starts speaking after the FIRST sentence
  memory         SQLite: messages, facts, rolling summary
```

The face: GNM head exported with 15 Oculus visemes + blink as morph targets
(authored via landmark-constrained least squares in GNM's expression space —
see tools/build_visemes.py), rendered in three.js.

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
| `JARVIS_CLOUDCLI_URL` | `http://127.0.0.1:3001` | CloudCLI server (or data/cloudcli.json) |
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
- If announcements stop, check the hook is still in ~/.claude/settings.json
  and `curl http://127.0.0.1:8710/api/health` returns ok.
