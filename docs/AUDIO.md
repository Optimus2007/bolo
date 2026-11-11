# Bolo audio data: datasheet, consent policy, pipeline

Two very different kinds of audio. They are never mixed in reporting.

| | Synthetic speech (TRAIN) | Human recordings (EVAL) |
|---|---|---|
| Source | Indic Parler-TTS reading the synthetic text dataset | Real people via `bolo record` |
| Used for | Whisper fine-tuning (train/dev), a synthetic test | **Testing only**: the headline speech benchmark. **Never used for training** (enforced in code: `whisper_ft.HumanAudioError`) |
| Where it lives | Kaggle/Colab working dir → `data/tts/` locally (gitignored) | `recordings/` on the laptop only (gitignored) |
| Committed to git | Never (only scripts that regenerate it) | Never |
| Licence | Generated with an Apache-2.0 model from our own text; shareable | Speakers' consent decides (see below) |

## 1. Human recordings (`bolo record`)

```
uv run bolo record            # opens http://127.0.0.1:8765/  (local only, not reachable from the network)
```

- **Share mode** (friends record from their own phone/laptop): `uv run bolo record --share` prints a private
  `https://….trycloudflare.com/?invite=…` link (free Cloudflare quick tunnel, no account; installed with
  `brew install cloudflared`). The server still binds to 127.0.0.1; the tunnel gives it HTTPS, which phones need for
  microphone access. Audio is sent over the encrypted link and **stored only on this laptop** in `recordings/`. It passes
  through Cloudflare's network (TLS terminates there) but is not stored by us anywhere else. Protection:
  - a random **invite code** per session is required on every request (no code → 403),
  - each speaker gets a **private key** at consent time, kept in their browser; uploads and *withdraw* need it, so nobody can
    record as, or delete, someone else,
  - a pseudonym that is already taken gets a new id (no takeover); at most 200 speakers per store,
  - the link **auto-closes** after `--minutes` (default 120) and dies when the command stops. Keep the laptop awake
    (`caffeinate -d`) while friends record.
  Rows recorded this way carry `"via": "link"`. The owner can always delete a speaker from the CLI (`Store.withdraw(owner=True)`).
- **Consent first.** Before the first recording a speaker picks a *pseudonym* and ticks two required boxes:
  being recorded, and use within the Bolo project (kept private). A third box, inclusion in a future
  **public** dataset (CC BY 4.0), is optional and off by default. Consent is stored per speaker in
  `recordings/speakers.json` and copied into every manifest row.
- **Withdraw** deletes that speaker's audio files, manifest rows and speaker entry. To withdraw for someone
  later: start `bolo record`, enter the same pseudonym (it maps to the same id), and press *Delete my recordings*.
  Or delete `recordings/audio/<id>/` and their rows by hand.
- **Minimal personal data.** Only a pseudonym and optional coarse fields (voice, self-described as female/male/other;
  age range; region; mic type). Voice is used only to check that accuracy is equal across groups. Group results are
  reported only when a group has at least 3 speakers (`MIN_SPEAKERS_PER_GROUP`), so no individual can be singled out.
  Do not type real names. Voice itself is personal data: treat every file as private.
- **Prompts** come only from the held-out **test** split (`recordings/prompts_v0.jsonl`, a pool of 120). Each speaker gets a
  short personal set: **40 read + 15 optional bonus** (`--n-read/--n-free`), balanced over all 9 intents and preferring
  prompts earlier speakers were not given. So ~4 minutes per person, and two speakers together cover all 80 read prompts:
  - `read` (80, shown first): read a test sentence (Roman shown, mixed-script hint below).
  - `free` (40, optional **bonus round** offered after all read prompts): only the goal is shown (“Call Ankita on WhatsApp”); the speaker says it in their own words.
    These measure real phrasing variation that no template covers. The label is the goal (intent + slots).
    A typed transcript is optional; without one, free prompts are scored end-to-end (speech → action) and not by WER.
- **Format:** browser-captured, resampled in JS to 16 kHz mono 16-bit WAV. Browser echo-cancellation,
  noise-suppression and auto-gain are switched **off** so the audio is the raw mic. 0.3–15 s per clip.
- **Manifest** `recordings/manifest.jsonl`, one row per utterance:
  `utt_id, speaker_id, prompt_id, kind, source_id, audio, sample_rate, duration_s, intent, slots,
  transcript, transcript_source, consent{record, project_use, public_release, given_at}, speaker_meta, mic, created, tool`.

### Rules (read before recording friends)

1. **Never commit or release anyone's audio without their explicit consent.** `recordings/` and all audio
   extensions are gitignored; keep it that way. Check `git status` before every commit.
2. Do not upload `recordings/` to Kaggle, Colab, Google Drive or anywhere else. Speech eval runs locally.
3. A future public release may include **only** speakers with `public_release: true`, re-confirmed with them
   at release time. Everyone else stays private, forever.
4. If someone asks to be removed, remove them (withdraw), including from any derived results tables that
   list per-speaker numbers.
5. Report only aggregate numbers (per-speaker breakdowns by pseudonym id at most).

## 2. Synthetic speech (training data)

- **Teacher:** `ai4bharat/indic-parler-tts` (Apache-2.0; gated, you must accept its conditions on Hugging Face).
  Its training data (per the model card) includes IndicTTS, LIMMITS, Rasa (CC BY 4.0) and GLOBE.
- **Code:** `src/bolo/audio/tts.py` (`plan` is tested locally; `synthesize` runs in `notebooks/01_tts_synth.ipynb`).
- **Voices split by speaker.** Train/dev: Rohit, Divya (Hindi); Thoma, Mary, Kabir, Sneha, Raghav (English).
  Test: Aman, Rani (Hindi); Priya, Vikas (English). No voice is in both.
- **Which text is spoken:** Hindi voices read the mixed-script `tts_text` (Hindi in Devanagari, English in
  Latin). English voices read the Roman `asr_text`, and only for English-heavy commands (en_ratio ≥ 0.5,
  with probability 0.35), where an English-accented speaker is realistic.
- **Descriptions** vary tone, pace, pitch and recording quality (clear / close / distant / light noise).
- **Post-processing:** resample to 16 kHz (FFT), trim leading/trailing silence, peak-normalise to 0.9.
  **Rejected** (logged in `rejected.jsonl`, not used): shorter than 0.4 s, longer than 12 s or 0.25 s per character
  (babbling), faster than 0.025 s per character, or more than 1% clipped samples. The target text is always
  `asr_text`, so a TTS mispronunciation becomes label noise. The notebook asks you to listen to samples.
- **Augmentation** (`src/bolo/audio/augment.py`, numpy-only, applied on the fly during training with p=0.8):
  speed perturbation 0.9/1.0/1.1 (p 0.6), synthetic room reverb RT60 0.15–0.7 s (p 0.5), additive
  noise at SNR 5–30 dB (p 0.8; white/pink/brown/mains hum, or real noise clips if supplied), phone band
  300–3400 Hz (p 0.2), gain −12…+3 dB with clip protection. Every op is seeded and returns its params.
  *Limitation:* there's no real noise corpus yet (MUSAN is about 11 GB; ESC-50 is NC-licensed), so the noise is synthetic.

## 3. ASR fine-tuning (`notebooks/02_whisper_finetune.ipynb`)

- **Base:** `Oriserve/Whisper-Hindi2Hinglish-Swift` (openai/whisper-base fine-tune, 72.6M, Apache-2.0,
  Roman-script output). Its card decodes with `language="en", task="transcribe"`. We train with the same prefix.
- **Target:** canonical Roman `asr_text` (lowercase, no punctuation). Scoring lowercases and strips punctuation
  from hypotheses.
- **Loop:** plain PyTorch (AdamW, lr 1e-5, warmup 50, cosine, fp16, bs 32, 5 epochs), best checkpoint by dev WER.
- **Baselines** on the same synthetic test: openai/whisper-base (hi, en), openai/whisper-small (hi),
  Swift (en, hi). Phase 5 repeats this on the human recordings.
- **Metrics** (`src/bolo/audio/metrics.py`): WER, CER, *phonetic WER* (word match after `phonetic_key`, which
  forgives Roman spelling variants such as kholo/kolo), sentence exact match. Devanagari hypotheses are
  transliterated to Roman first. This still penalises English loanwords written in Devanagari
  (टर्मिनल → "tarminal").

### Local smoke test (done, 2026-11-11)

16 clips made with macOS `say` (voices Lekha hi_IN, Aman and Tara en_IN) from test sentences. On these, Swift
baseline: WER 0.34 / phonetic WER 0.29 (en prefix). 10 training steps on MPS ran, saved and re-evaluated.
**These are pipeline checks only, not results** (16 clips, robotic voices, trained on the eval clips).
Example of the expected gap: "claude code start karo" → "Close codes dot karo."

## 4. Running it (Kaggle, free tier)

1. `uv run python scripts/make_kaggle_bundle.py` → `data/kaggle/bolo-bundle.zip` (code + synthetic text only,
   0.6 MB). Kaggle → Datasets → New → upload the zip → **Private** → name `bolo-bundle`.
2. Hugging Face: accept the conditions of `ai4bharat/indic-parler-tts`, create a **read** token.
   Kaggle notebook → Add-ons → Secrets → `HF_TOKEN`.
3. Notebook 01 (upload `notebooks/01_tts_synth.ipynb`): add input `bolo-bundle`, GPU T4, Internet on.
   Try `N = {"train": 64, "dev": 16, "test": 32}` first, check the samples by ear, then the full run with Save & Run All.
4. Notebook 02: add inputs `bolo-bundle` + the output of notebook 01. GPU T4, Internet on, Save & Run All.
5. Download `bolo-whisper/` → `models/bolo-whisper/` and `results_asr.json` → `results/asr/`.
   Colab works too: upload/unzip `bolo-bundle.zip` into `/content`, add `HF_TOKEN` in Colab secrets.
