# ChopShop — Overnight Build Plan

> **For the Claude Code session running this:** You are building ChopShop, an AI-driven sampling instrument. A person drops in a song, and an AI (talking to ChopShop over MCP) splits it into stems, chops it into pads, adds swing, pitch-corrects vocals, and renders the result, while the person directs by describing what they want.
>
> This is an unattended overnight run. Nobody will answer questions. Work through the phases in order. **Every phase ends in a verification gate with measurable pass criteria. A phase is done only when its gate passes and the evidence is written to `PROGRESS.md`.** If a gate cannot pass, follow the "When stuck" rules. Never weaken a threshold to make a gate pass.

---

## 0. Ground rules

1. **Evidence over claims.** Every gate result in `PROGRESS.md` must include the exact command run and the relevant numeric output: test counts, measured values, and thresholds.
2. **Commit after each passing gate** with a message like `gate G3: slicing passes (12/12 tests)`. Commit work in progress before starting risky refactors.
3. **Never weaken thresholds.** If a threshold seems wrong, write down why in `PROGRESS.md` under "Threshold disputes" with measurements, keep the original gate marked FAILED, and move on. A human will decide in the morning.
4. **No copyrighted audio.** All test audio is synthesized by the project itself (Phase 2). Do not download songs, music videos, or datasets of commercial music.
5. **Headless environment.** This container has no sound card and no MIDI devices. Everything must work and be tested **offline** (render to WAV, analyze the WAV). Live playback and MIDI are compile-only stretch goals behind a cargo feature.
6. **Final target is an Apple Silicon Mac** (M2, macOS). Avoid Linux-only dependencies. Anything platform-specific goes behind a feature flag, with Mac setup documented in the README.
7. **When stuck** (same failure after 3 materially different attempts, or more than about 45 minutes on one problem):
   - Write the blocker to `PROGRESS.md` with error output and what you tried
   - Put the blocked feature behind a cargo feature flag or a fallback backend
   - Continue to the next phase. Don't burn the night on one wall.
8. **Read the docs for the version you install.** Several crates here have had breaking API changes between versions (notably `rmcp`, which is pre-1.0, and `rubato`, which reached 1.x). After `cargo add`, check docs.rs for that exact version before writing code against it.

---

## 1. What we're building

```
┌────────────────────┐   MCP (stdio)   ┌────────────────────────────────────┐
│ Claude (Desktop /  │◄───────────────►│ chopshop-mcp (Rust, rmcp)          │
│ Claude Code)       │                 │  tools: load, separate, slice,     │
└────────────────────┘                 │  pads, swing, pitch, autotune,     │
                                       │  analyze, render, undo …           │
                                       └──────────────┬─────────────────────┘
                                                      │ Rust API
            ┌─────────────────────────────────────────┼────────────────────────────┐
            ▼                        ▼                ▼                            ▼
   ┌────────────────┐    ┌───────────────────┐  ┌──────────────┐        ┌───────────────────┐
   │ chopshop-core  │    │ chopshop-analysis │  │ chopshop-sep │        │ chopshop-sidecar  │
   │ buffers, IO,   │    │ onsets, tempo,    │  │ StemSeparator│◄──────►│ (Python, uv)      │
   │ session model, │    │ pitch (YIN), RMS, │  │ trait +      │ JSON-  │ librosa: onsets,  │
   │ render engine, │    │ centroid, SDR     │  │ backends     │ RPC    │ pyin, autotune,   │
   │ DSP, undo log  │    │                   │  │              │ stdio  │ HPSS, separation  │
   └────────────────┘    └───────────────────┘  └──────────────┘        └───────────────────┘
            ▲
   ┌────────┴────────┐       ┌────────────────────────────┐
   │ chopshop-cli    │       │ chopshop-live (feature)    │
   │ every op as a   │       │ cpal playback + midir pads │
   │ subcommand      │       │ compile-only tonight       │
   └─────────────────┘       └────────────────────────────┘
```

### Key architecture decisions (already researched, so use these)

| Concern | Choice | Why |
|---|---|---|
| MCP server | **`rmcp`**, the official Rust MCP SDK (tokio), with the `server`, `macros`, `transport-io`, and `schemars` features | Official SDK with `#[tool]` macros and JSON Schema generation for tool params. It's pre-1.0, so pin the version and read docs.rs for it. |
| Decode | **`symphonia`** | Pure Rust; handles WAV, FLAC, MP3, AAC, and OGG |
| WAV write | **`hound`** | Simple and reliable |
| Resample | **`rubato`** | Standard Rust resampler. The 1.x API differs from older tutorials, so check the docs. |
| FFT | **`rustfft`** / **`realfft`** | For spectral flux onset detection, the spectral centroid, and STFT |
| Pitch shift / time stretch | **`signalsmith-stretch`** crate (wraps the MIT-licensed Signalsmith Stretch C++ library) | High quality; supports both transpose-in-semitones and time stretch. Needs a C++ compiler at build time; if it won't build, `ssstretch` is an alternative binding, with sidecar `librosa.effects` as the last-resort fallback. |
| Filters / EQ | **`biquad`** crate, or hand-rolled RBJ biquads | Simple, testable |
| Onset detection | **Native Rust spectral flux** in `chopshop-analysis`, cross-validated against librosa | Avoid `aubio-rs`: it's GPL-3.0 and unmaintained since 2021 |
| Pitch detection | **Native Rust YIN** for tests and analysis; **`librosa.pyin`** in the sidecar for autotune | YIN is ~100 lines and easy to verify on sines |
| Autotune | **Sidecar: `librosa.pyin` → snap f0 to scale → `psola.vocode`** | A known working recipe; the `psola` package is on PyPI |
| Stem separation | **`StemSeparator` trait with 3 backends:** (1) `OnnxDemucs` via the **`stem-splitter-core`** crate (pure Rust, ONNX Runtime, htdemucs, 4 stems; downloads a ~200 MB model on first use), (2) `SidecarSeparator` via the **`audio-separator`** Python package, (3) `HpssFallback` via `librosa.effects.hpss`, which needs no model | Model downloads may be blocked by this container's network allowlist (see Phase 0). The HPSS fallback guarantees the pipeline works end to end regardless. |
| Python interop | **Long-lived subprocess sidecar speaking newline-delimited JSON-RPC over stdio**, managed with `uv`. Pass **file paths**, not sample arrays. | The Rust binaries still build and run without Python, a sidecar crash can't take down the MCP server, it's easy to test, and librosa's import cost is paid once per session. Not PyO3. |
| State | `Session` struct serialized to `session.json` in a project dir; every mutating op appends to an **undo log** | Reversible actions are what make conversational editing safe |
| Test oracle | **Synthetic audio with known ground truth** (Phase 2) | Lets every DSP feature be verified numerically with no copyrighted material |
| MCP verification | **MCP Inspector CLI mode** (`npx @modelcontextprotocol/inspector --cli … --method tools/call`) **plus** a Rust integration test using `rmcp`'s client feature (`transport-child-process`) | The Inspector CLI is non-interactive and exits non-zero when a tool returns `isError`, so it chains cleanly in scripts |

---

## 2. Repository layout

```
chopshop/
  Cargo.toml                 # workspace
  crates/
    chopshop-core/           # AudioBuffer, IO, Session, render, DSP, undo
    chopshop-analysis/       # onsets, tempo, YIN, RMS/LUFS-ish, centroid, SDR
    chopshop-sep/            # StemSeparator trait + backends
    chopshop-sidecar-client/ # Rust side of the JSON-RPC sidecar protocol
    chopshop-mcp/            # rmcp server binary
    chopshop-cli/            # CLI binary (also used by gate scripts)
    chopshop-testgen/        # synthetic corpus generator (binary + lib)
    chopshop-live/           # feature-gated cpal + midir (stretch)
  sidecar/
    pyproject.toml           # uv project: librosa, numpy, soundfile, psola, (optional) audio-separator
    chopshop_sidecar/
      server.py              # JSON-RPC loop
      ops/onsets.py, pitch.py, autotune.py, hpss.py, separate.py
    tests/                   # pytest
  gates/
    run_all.sh               # runs every gate, prints a summary table, nonzero on any failure
    g0_env.sh … g12_review.sh
  corpus/                    # generated, gitignored except metadata schema
  PROGRESS.md                # gate log (append-only during the run)
  README.md                  # Mac setup, Claude Desktop/Code MCP config, usage
  DEMO.md                    # transcript of an end-to-end MCP session
  LICENSES.md                # every dependency/model license that matters
```

---

## 3. Phases and gates

Each phase lists **build** steps, then a **gate**. Gate scripts live in `gates/` and must be runnable non-interactively. `gates/run_all.sh` must re-run every gate from scratch at the end.

### Phase 0: Environment probe (G0)

**Build**
- Check for and record versions: `rustc`, `cargo`, `python3`, `uv` (install with pip if missing), `node`/`npx`, a C++ compiler and `libclang` (signalsmith-stretch builds C++ and generates bindings), and `ffmpeg` (optional). Install anything missing via apt or pip where possible.
- **Network probe.** Try reaching `crates.io`, `pypi.org`, `github.com`, `huggingface.co`, and the URL `stem-splitter-core` uses for its model download (find it in the crate source). Record which succeed. This environment uses a domain allowlist, so some will fail. That's expected and is why fallbacks exist.
- Write `ENV_REPORT.md` with the results.

**Gate G0 (pass when all hold)**
- `ENV_REPORT.md` exists and lists every tool's version, or MISSING
- The network table is filled in
- A decision is written: which separation backends are viable tonight

### Phase 1: Workspace skeleton and audio IO (G1)

**Build**
- Cargo workspace with all crates. CI-style checks wired into `gates/g1_io.sh`.
- `chopshop-core::AudioBuffer { sample_rate: u32, channels: Vec<Vec<f32>> }` with helpers: mono mixdown, duration, slicing by sample range, gain, and fades (equal-power and linear).
- `io::load(path)` via symphonia (any supported format → f32 planar); `io::save_wav(path, buf, bit_depth)` via hound.
- `resample(buf, target_sr)` via rubato.

**Gate G1**
- `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` all pass
- WAV round trip: 32-bit float save→load max abs error **≤ 1e-7**; 16-bit max abs error **≤ 1/32768**
- Resample round trip: 1 kHz sine at 44.1 kHz → 48 kHz → 44.1 kHz, SNR **≥ 60 dB** (ignore the first and last 1,024 samples)
- Loading a 10-second stereo file reports the correct duration to ±1 ms

### Phase 2: Synthetic ground-truth corpus (G2)

This corpus is the oracle for every later gate, so make it deterministic (seeded RNG) and well documented.

**Build** `chopshop-testgen`, which writes `corpus/<name>/{mix.wav, stems/*.wav, truth.json}`:
- **drums.wav**: kicks (exponential sine sweep 150→50 Hz, 200 ms decay), snares (band-passed noise plus a 180 Hz body, 120 ms), and closed hats (high-passed noise, 40 ms) on a grid at a known BPM (default 92). `truth.json` lists every hit time, type, and velocity.
- **bass.wav**: low notes (40–110 Hz) with known MIDI pitches and on/off times, using a sine plus 2nd/3rd harmonics with an ADSR envelope.
- **vocal.wav**: a pseudo-voice made from a band-limited sawtooth through 3 formant band-pass filters, with 5.5 Hz vibrato (±20 cents), following a melody in C major. Generate **two versions**: `vocal_in_tune.wav` and `vocal_detuned.wav`, offset by a known amount (default **+35 cents**). Truth includes the frame-level target f0.
- **pad.wav**: sustained chords (other/harmony stem).
- **mix.wav** = the sum of the stems, normalized to -1 dBFS peak. Record the normalization gain in the truth file so stems can be compared.
- Generate at least **3 corpora** with different BPMs (80, 92, 128), seeds, and keys.
- **Copyright-safe by construction:** document that in README.

**Gate G2**
- Running `chopshop-testgen --all` twice produces **byte-identical** files (determinism)
- `truth.json` validates against a schema (write one; check it in a test)
- Sum-of-stems equals the mix to within **1e-6** max abs error, accounting for the recorded gain
- The pytest side can load every file and its truth (catches format mismatches early)

### Phase 3: Analysis, the AI's "ears" (G3)

**Build** `chopshop-analysis`:
- `onsets(buf) -> Vec<f64 seconds>`: spectral flux (log-magnitude STFT, half-wave rectified difference, adaptive median threshold, peak picking with a minimum inter-onset interval)
- `tempo(onsets or buf) -> bpm` (autocorrelation of the onset envelope, with octave-error handling)
- `yin(buf, fmin, fmax) -> Vec<Option<f64>>` frame-wise f0
- `rms_db`, `peak_db`, `crest_factor`, `spectral_centroid_hz`, and simple `loudness_lufs_approx` (K-weighted RMS is fine; document that it's approximate)
- `sdr(estimate, reference) -> dB` and `sdr_improvement(estimate, reference, mixture)`
- `describe(buf) -> AnalysisReport` (serde-serializable). This is what the MCP `analyze` tool returns, so make it concise and human-meaningful.

**Sidecar** (start it in this phase): JSON-RPC methods `ping`, `onsets` (librosa), `pyin`, `tempo`.

**Gate G3**
- **Onsets (Rust)** on every `drums.wav`: F-measure **≥ 0.95** with a ±50 ms tolerance window against `truth.json`
- **Rust vs librosa agreement** on the same files: F-measure **≥ 0.90** between the two detectors (±50 ms)
- **Tempo** within **±2 BPM** of truth on all 3 corpora (allowing half/double-tempo answers only if you also report the octave-corrected value and it's within ±2)
- **YIN** on pure sines at 110, 220, 440, and 880 Hz: median error **< 0.5%**
- **YIN on `vocal_in_tune.wav`**: median abs deviation from the truth f0 **< 15 cents** in voiced frames
- `sdr(x, x)` is very high (≥ 100 dB or +∞, handled), and `sdr(noise, x)` is ≤ 0 dB
- **Performance (release build):** onsets on a 3-minute stereo 44.1 kHz file in **< 1.0 s**

### Phase 4: Session model, slicing, pads, undo (G4)

**Build**
- `Session` holds sources, stems, slices (source id, start/end samples, fade-in/out), up to 16+ pads (pad → slice, gain, pitch semitones, stretch ratio), a groove (swing amount 0–1, resolution 1/8 or 1/16), tempo, and an undo stack.
- Slicing modes: `onsets` (cut at detected onsets, optional sensitivity), `grid` (every N beats at session tempo), `equal` (N equal slices), and `manual` (list of times). Apply short fades (default 2 ms) to avoid clicks.
- `assign_pads(slice_ids, start_pad)`.
- Every mutating op is a `Command` with `apply`/`invert`. `undo()` pops and inverts. A session content hash (stable over serde JSON) is used for testing.
- Session persistence: `session.json` plus audio files in a project directory.

**Gate G4**
- Onset slicing of each `drums.wav` yields slice starts matching truth hits (F ≥ 0.95, ±50 ms)
- **Lossless reconstruction:** with fades disabled, concatenating `equal`/`grid` slices reproduces the source with max abs error **< 1e-6**
- **Click-free:** with default fades, no slice boundary has a sample-to-sample jump larger than **0.05** in rendered output, and the first/last sample of every faded slice is **< 1e-3** in absolute value
- **Undo:** for every mutating command type, a property test (proptest or quickcheck, ≥ 200 random cases) shows that `apply` then `undo` returns the session hash to its prior value
- Save → load round trip preserves the session hash

### Phase 5: Render engine and groove (G5)

**Build**
- `Pattern` is a list of steps (pad, step index, velocity) at a resolution, plus a length in bars.
- `render(session, pattern) -> AudioBuffer`: offline, deterministic, with per-pad gain, pitch/stretch applied (cached), and voice summing with a soft limiter to avoid clipping.
- **Swing**: delay every off-beat step (odd 16ths at 1/16 resolution, or odd 8ths at 1/8) by `swing * step_duration / 2`. Document the formula: swing 0.0 is straight, 1.0 delays the off-beat a full half step (triplet-ish feel around 0.66).

**Timing-precision warning:** a standard onset detector with a 512-sample hop only resolves about 11.6 ms at 44.1 kHz, which is too coarse for the ±2 ms checks below. For timing gates, use a fine-resolution mode: hop ≤ 64 samples, or locate each hit by the first sample where the envelope crosses a threshold. Also cross-check against the render engine's own event log (scheduled trigger times), so the gate verifies both the scheduler and the audio.

**Gate G5**
- Render 4 bars of straight 16ths of a kick slice at 92 BPM with swing 0.0, detect onsets, and confirm every hit is within **±2 ms** of the grid
- Same with swing 0.6: on-beat hits within **±2 ms**, off-beat hits delayed by `0.6 × step/2` within **±3 ms**
- Render is **deterministic**: two renders are byte-identical
- **No clipping:** a 16-voice pile-up stays at peak ≤ 0 dBFS
- **Performance:** rendering 16 bars with 8 active pads takes **< 0.5 s** in release mode

### Phase 6: Pitch and time (G6)

**Build**
- `pitch_shift(buf, semitones)` and `time_stretch(buf, ratio)` via signalsmith-stretch (fall back to `ssstretch`, then sidecar librosa, per the decision table). Expose both on pads.

**Gate G6**
- A 440 Hz sine shifted **+3 semitones** measures **523.25 Hz ±0.5%** (YIN median), with duration unchanged ±1%
- Shifted **−12 semitones**: 220 Hz ±0.5%
- Time stretch ×1.5: length = 1.5 × input ±1%, pitch unchanged ±0.5%
- Time stretch ×0.75 on `drums.wav`: onsets scale by 0.75 (F ≥ 0.9 against scaled truth, ±50 ms)

### Phase 7: Autotune (G7)

**Build** (sidecar): `autotune(path, key, scale, strength 0–1, retune_speed_ms) -> path`, implemented as `librosa.pyin` → snap each voiced frame's f0 to the nearest scale degree (strength blends original and snapped f0; retune speed smooths the target with a one-pole filter) → `psola.vocode`. Expose this through the Rust client and as a pad/slice operation.

**Gate G7**
- On `vocal_detuned.wav` (+35 cents) with key C major and strength 1.0: after autotune, the median abs deviation from the nearest C-major note in voiced frames is **< 10 cents**, down from about 35 before (report both numbers)
- Strength 0.0 output stays within **3 cents** of the input pitch (it's a no-op on pitch)
- Output duration equals input ±1%
- Silence stays silent: in truth-silent regions, output RMS stays below **−60 dBFS** (compare absolute levels; a dB ratio against near-zero input is meaningless)
- Voiced regions keep their level: output RMS in voiced regions is within **±3 dB** of the input

### Phase 8: Stem separation (G8)

**Build**
- The `StemSeparator` trait has `separate(path, out_dir) -> StemSet { vocals, drums, bass, other: Option<path> }` and `name()`, with backends chosen by availability (from the G0 decision) and configurable.
- `OnnxDemucs` uses `stem-splitter-core`, behind cargo feature `onnx`. Cache the model in a configurable dir. If the download is blocked, the backend reports *unavailable* cleanly, without panicking.
- `SidecarSeparator` uses `audio-separator`. It's optional too; it may also be blocked by model downloads.
- `HpssFallback` uses `librosa.effects.hpss` → percussive = "drums," harmonic = "other," plus a low-pass of harmonic below ~150 Hz = rough "bass." It's crude, but it always works.

**Gate G8** (per available backend, run on all 3 corpora; report a table of SDR and SDRi per stem)
- **HPSS fallback:** drums SDRi **≥ 3 dB**, and the pipeline completes on all corpora
- **Demucs (if available):** drums and bass SDRi **≥ 3 dB**. Record vocals SDRi but don't gate on it, since the synthetic pseudo-voice is far from Demucs's training data. Note this in PROGRESS.
- **Graceful degradation:** with the network disabled (or model dir empty and download forced to fail), `separate` falls back to HPSS automatically and says so in its result. Write a test for this.
- **Performance:** record wall-clock time per backend for a 30 s mix (informational only, not gated)

### Phase 9: MCP server (G9)

**Build** `chopshop-mcp`, an rmcp stdio server. Every tool has a JSON Schema for its params, a clear description written for an AI caller, and returns structured, concise results. **Errors are returned as tool errors (`isError: true`) with an actionable message, never as panics.** No `unwrap()`/`expect()` in handler paths (enforce with a clippy lint or a grep check in the gate).

Tools (minimum set):
| Tool | Purpose |
|---|---|
| `project_open` / `project_new` | Choose a project dir; load or create a session |
| `load_audio` | Import a file as a source (any format symphonia decodes) |
| `separate_stems` | Run the best available backend and return stem ids, plus which backend was used |
| `slice` | Mode `onsets` / `grid` / `equal` / `manual` with options; returns slice ids with times |
| `assign_pads` | Map slices to pads |
| `set_pad` | Gain, pitch semitones, stretch ratio, reverse, fades |
| `autotune` | On a stem or slice: key, scale, strength, retune speed |
| `set_groove` | Swing amount and resolution; tempo |
| `write_pattern` | Steps for a pattern (simple text grid like `"x...x...x...x..."` per pad is welcome; it's AI-friendly) |
| `render` | Render a pattern to WAV; returns the path plus an `analyze` summary |
| `analyze` | The "ears": AnalysisReport for any source, stem, slice, or render |
| `describe_session` | Compact text and JSON overview of everything |
| `undo` / `history` | Reversibility |

Also expose a **`vibe_presets`** resource or tool: a small, documented table mapping producer words to concrete parameter moves, such as "punchier" → shorter fades and +2 dB pad gain on drums, "tha-thunk / more swing" → swing +0.15, "darker" → low-pass on the pad, "airier vocal" → high-shelf on the vocal stem. Keep it honest: these are starting points the AI can apply and then verify with `analyze`.

**Gate G9**
- `npx @modelcontextprotocol/inspector --cli <chopshop-mcp binary> --method tools/list` lists **every tool above** with non-empty descriptions and schemas (assert this with a script parsing the JSON output)
- **Scripted end-to-end via Inspector CLI** (`gates/g9_mcp.sh`), with every call exiting 0: `project_new → load_audio(corpus mix) → separate_stems → slice(onsets, drums stem) → assign_pads → set_groove(swing 0.6) → write_pattern → render → analyze`. Assert the render exists, has the expected duration ±1%, and the analyze report's onset count is within ±10% of the pattern's hit count.
- **Error handling:** calling `slice` on a nonexistent id returns `isError: true` with a helpful message, and the Inspector CLI exits non-zero as expected. The server keeps running afterward (the next call succeeds).
- **Rust integration test** using rmcp client + `transport-child-process` covers the same happy path
- **Undo over MCP:** after the e2e run, N `undo` calls restore the initial session hash

### Phase 10: CLI, docs, and demo (G10)

**Build**
- `chopshop-cli` mirrors the MCP tools as subcommands (useful for humans and debugging).
- **README.md**: what it is; **Mac setup from zero** (rustup, uv, `cargo build --release`, sidecar `uv sync`); how to register the MCP server with **Claude Desktop** (`claude_desktop_config.json` snippet) and **Claude Code** (`claude mcp add chopshop -- /path/to/chopshop-mcp`); backend notes (Demucs model download, HPSS fallback); and troubleshooting.
- **DEMO.md**: a realistic transcript of a conversation driving the tools ("give me just the drums," "chop them on the hits," "put them on pads 1–8," "add some swing," "render 4 bars"), with the actual tool calls and real outputs produced by running them against a corpus file.
- **LICENSES.md**: licenses for all crates, Python packages, and models used (check Demucs weights, signalsmith-stretch, stem-splitter-core, audio-separator, and the UVR models; flag anything non-permissive).

**Gate G10**
- Every command in README's "quick start" runs successfully in this container, except the Mac-only steps, which must be clearly labeled. Test by copying commands into a script.
- Every tool call in DEMO.md was actually executed (keep the script that produced it in `gates/`)
- LICENSES.md has no "unknown" entries for anything that ships in default features

### Phase 11: Full regression (G11)

- `gates/run_all.sh` from a **fresh clone** (`git clone . /tmp/chopshop-fresh && cd /tmp/chopshop-fresh`) runs every gate G0–G10 and prints a summary table (gate, status, key metric, threshold)
- Gate G11 passes when the script exits 0, or when every failing gate has a documented entry under "Threshold disputes" or "Blockers" in PROGRESS.md

### Phase 12: Independent review (G12)

Spawn a **fresh subagent** that hasn't seen your work. Give it only this plan and the repo path, and have it:
1. Run `gates/run_all.sh` itself from a fresh clone
2. Spot-check at least 3 gates by reading the gate script and confirming it actually measures what this plan says (no tautological tests, no loosened thresholds, no skipped assertions)
3. Check for `unwrap`/`expect` in MCP handler paths
4. Write findings to `REVIEW.md`

**Gate G12:** `REVIEW.md` exists, and every issue it raises is either fixed (re-run the affected gate) or listed in PROGRESS.md with a reason.

---

## 4. Stretch goals (only after G12)

Each still needs a test and an entry in PROGRESS.md.
1. **Effects chain** on pads and stems: biquad EQ (low/high shelf, peak, LP/HP), a simple compressor, and a Freeverb-style reverb. Verify with analysis: an HP at 200 Hz reduces energy below 100 Hz by ≥ 20 dB on white noise, and the compressor reduces crest factor on drums.
2. **`chopshop-live`** (feature `live`): cpal playback of pads triggered from the computer keyboard (QWERTY rows → 16 pads), plus `midir` input for a Novation Launchpad or Akai MPK Mini (note-on → pad). Compile-only here, so `cargo build --features live` must pass. Document manual test steps for the Mac in README.
3. **Key detection** (chroma template matching) in `analyze`, so autotune can default to the detected key.
4. **Reference-match "ears"**: `compare(a, b)` returns differences in loudness, brightness, punch (crest), and swing, so the AI can say "your render is darker and less punchy than the reference" and act on it.

---

## 5. PROGRESS.md format

```markdown
# ChopShop progress

## Gate summary
| Gate | Status | Key metric | Threshold | Commit |
|---|---|---|---|---|
| G0 | PASS | 4/5 domains reachable; HF blocked | n/a | abc123 |
| G3 | PASS | onset F=0.98 / tempo err 0.4 BPM | F≥0.95 / ±2 | def456 |

## Log
### [time] G3 attempt 2
Command: `gates/g3_analysis.sh`
Output (trimmed): …
Result: PASS

## Blockers
## Threshold disputes
## Notes for the human
```

Keep "Notes for the human" short and useful: what works, what doesn't, the 3 most interesting things you learned, and the exact commands to try it on a Mac.

---

## 6. Definition of done (for tonight)

- G0–G12 all PASS, or each failure documented with evidence and a reason
- `cargo build --release` produces `chopshop-mcp` and `chopshop-cli`
- A person on a Mac can follow README, register the MCP server with Claude, and have a conversation that splits a song, chops the drums onto pads, adds swing, autotunes a vocal, and renders a loop