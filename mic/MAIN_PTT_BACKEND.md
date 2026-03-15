# main.py Push-to-Talk Backend

This document explains how [main.py](main.py) works for UDP-driven push-to-talk (PTT) transcription.

## What it does
- Listens on UDP `0.0.0.0:5005` for PTT control packets.
- Records microphone audio locally on laptop while PTT is active.
- Transcribes locally using `faster-whisper` (no cloud STT).
- Prints transcript text to terminal.

## Startup flow
1. `run_server()` loads Whisper once via `load_model_once()`.
2. Tries GPU (`cuda`, `float16`), falls back to CPU (`int8`).
3. Creates a shared `PyAudio` instance and capture state.
4. Binds UDP socket and enters infinite packet loop.

## UDP packet protocol
Start packets:
- `LISTEN`
- `START`
- `PTT_DOWN`
- `PRESS`

Stop packets:
- `STOP`
- `PTT_UP`
- `RELEASE`

Compatibility behavior:
- If `LISTEN` arrives while already recording, it is treated as stop + transcribe.

## Recording state machine
- `start_recording(...)`:
- No-op if already recording (debounce-safe).
- Opens default input mic stream.
- Starts `capture_loop(...)` thread to append PCM frames.
- `stop_recording(...)`:
- No-op if not recording.
- Signals capture thread to stop.
- Closes stream and returns concatenated bytes.

## Transcription path
1. Raw PCM16 bytes are converted to float32 in `bytes_to_float32_mono(...)`.
2. `transcribe_local(...)` runs Whisper with:
- `language="en"`
- `vad_filter=True`
- `beam_size=5`
- `temperature=0.0`
3. Segment text is joined and printed as `Transcribed: ...`.

## Error handling and shutdown
- Mic open/read failures are caught and printed.
- UDP socket errors are caught and printed.
- `Ctrl+C` exits cleanly.
- `finally` always stops any active recording and terminates `PyAudio`.

## How to run
From repository root:

```bash
python mic/main.py
```

Then send UDP start/stop packets from ESP32 to port `5005`.
