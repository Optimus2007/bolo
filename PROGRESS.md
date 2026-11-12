# PROGRESS

_Last updated: 2026-11-11 (Phase 5 done on synthetic and public speech; real command recordings pending)_

## Done
- Phase 0: private GitHub repo, .gitignore, folder structure, first commit.

- Phase 1: RESEARCH.md written (resources, licenses, gap analysis, plan + estimates).

- Phase 2 (templates): `configs/{schema,lexicon,templates}.yaml`, `src/bolo/{config,schema,splits,generator,cli}.py`,
  `bolo generate` / `bolo stats`, 22 tests passing, `docs/DATASHEET.md`, sample at `samples/text_v0_sample.jsonl`.
  v0 = 5000/750/1000 rows, ~14% `none`, 3 scripts, generated in <1 s.

- Phase 3: normalizer (`normalize.py`), eval harness (`evaluate.py`), rule baseline, Bolo-tiny (PhonoHash, from
  scratch, 0.69M params), MuRIL joint tagger, Qwen2.5-0.5B LoRA (MLX 4-bit), environment grounding
  (`grounding.py`), Bolo-tiny ONNX int8 export (0.8 MB, no accuracy loss), LLM paraphrases (train + reviewed test).
  Results in `results/RESULTS.md`. 56 tests passing.

- Phase 4 (code ready; heavy runs are on the user): `bolo record` (local consent-first recorder, `src/bolo/record/`),
  numpy audio augmentation (`src/bolo/audio/augment.py`), ASR metrics incl. phonetic WER (`audio/metrics.py`),
  TTS planning/synthesis (`audio/tts.py`), Whisper fine-tune/eval loop (`asr/whisper_ft.py`),
  `scripts/make_kaggle_bundle.py`, `scripts/make_notebooks.py` → `notebooks/01_tts_synth.ipynb`,
  `notebooks/02_whisper_finetune.ipynb`, `docs/AUDIO.md` (datasheet + consent policy). 71 tests passing.
  Local smoke test of the Whisper pipeline on MPS with 16 macOS `say` clips passed (pipeline check only).

- Phase 5 (started): `bolo eval` (`src/bolo/speech_eval.py`): ASR WER/CER/phonetic WER, NLU on ASR output
  (strict + grounded = end-to-end action), oracle-transcript row, latency/RTF, per set (never pooled).
  `scripts/make_say_eval.py` = 80 macOS `say` clips (4 voices) as a second, unseen-TTS synthetic set.
  First results: `results/speech/say_v0.md`. `scripts/kaggle_push.py` pushes the bundle and notebooks to Kaggle (private).
  The private dataset `optimus2005/bolo-bundle` is uploaded and the private kernel `bolo-tts-synth` is pushed.

- Phase 5 results (`results/SPEECH_RESULTS.md`): TTS run on Kaggle produced 4,922/289/977 clips. Bolo Sun (Swift fine tuned
  for 18 min on a T4) gets synthetic test WER 10.6% vs Swift 29.5%. On macOS voices (an unseen synthesiser) it gets WER 13.3% vs 28.8%,
  and E2E action 82.5% vs 62.5% (oracle 95%). On real OpenSLR-104 speech it is slightly worse (59.1% vs 55.2%): a
  specialisation trade-off. A WiSE-FT 50/50 blend gets 56.0% there but 78.8% E2E. Default = full fine tune. The model is in
  models/bolo-whisper (gitignored), the blend in models/bolo-whisper-wise050.
- Overnight 2026-11-11: Python API `bolo.pipeline.Bolo` + `bolo understand` + read-only `scan_mac`; packaging (LICENSE,
  configs in the wheel, v0.1.0, `speech` extra; the PyPI name `bolo` was free when checked); README rewritten in plain language;
  model cards for Bolo Tiny and Bolo Sun.

- Release prep (not published): `scripts/prepare_release.py` stages bolo-sun (fp16), bolo-tiny (safetensors + int8 ONNX)
  and the bolocmd-v0 dataset with HF model cards in data/release/ (gitignored); `docs/RELEASE_CHECKLIST.md` lists the owner-only steps.

## In progress
- Nothing running. All Kaggle jobs (TTS, fine tune, ablation) finished and their results are in results/.
- Ablation (2026-11-12): frozen encoder and 2 epoch variants forget slightly less (OpenSLR 57.2% / 57.5% vs 59.1%) but
  lose command accuracy (macOS voices 16.2% / 16.4% vs 13.3% WER, 76.3% vs 82.5% actions). Full fine tune stays default.
  The variant weights and the OpenSLR download were deleted afterwards.

- Decoding (2026-11-12): beam 5 + N-best rerank by NLU and grounding is the default in Bolo.listen. Held-out macOS
  set (240): correct action 73.8% -> 82.1%, WER 16.0% -> 12.5%, median 107 -> 177 ms. Hotword boosting did not help.
  bolo.adapt.Memory (aliases, usage prior, bandit weights) is opt in; simulation showed no gain beyond noise.

## Next
- Measure decoding + Memory on real recordings; tune rerank weights on public real speech, not on friends' audio.
- Training side from the research report: replay of pseudo-labelled general Hinglish + KL to base, real RIR/MUSAN noise,
  more varied Parler voices (Kaggle).
- Real command recordings from friends (10 people, 40 prompts each), then `bolo eval` on them: the headline numbers.
- Phase 6 (needs approval): Myna's voice (Bolo-Awaaz) and wake word. Phase 7: owner publishes per docs/RELEASE_CHECKLIST.md.
- Optional: regenerate paraphrases with an Apache-2.0 model (Qwen2.5-1.5B or 7B) for a releasable dataset.
- Tune the grounding threshold on real recordings.

## How to resume
```
cd ~/Documents/bolo && uv venv --python 3.12 .venv && uv pip install -e ".[dev,train,speech]"
cp scripts/pre-commit .git/hooks/ && chmod +x .git/hooks/pre-commit      # safety lock (after a fresh clone)
.venv/bin/pytest -q && .venv/bin/bolo generate --out data/text/v0 --sample-out samples/text_v0_sample.jsonl
.venv/bin/bolo understand "volume 40 pe kar do"                          # text -> action
.venv/bin/bolo understand some.wav --audio --ground                      # speech -> action (uses models/bolo-whisper-fp16)
caffeinate -d & .venv/bin/bolo record --share                            # friends record via a private link
.venv/bin/bolo eval --manifest recordings/manifest.jsonl say=data/say/manifest.jsonl --asr swift,models/bolo-whisper-fp16 --nlu rules,tiny --out results/speech/human_v1.json
python scripts/kaggle_push.py status | fetch ft | abl                   # Kaggle (private)
python scripts/prepare_release.py                                        # stage release files (no publishing)
```
Models not in git (regenerate or re-download): models/bolo-whisper (Kaggle output of bolo-whisper-ft), models/bolo-whisper-fp16
(`python scripts/to_fp16.py models/bolo-whisper models/bolo-whisper-fp16`), models/bolo-tiny (scripts/train_tiny.py).

## Decisions
- Layout: `src/bolo` (package), `configs/` (YAML schema), `scripts/`, `notebooks/` (Kaggle/Colab),
  `tests/`, `docs/`, `samples/` (small committed examples). `data/` and `models/` are gitignored.
- .gitignore blocks all audio formats, weights (safetensors/gguf/pt/npz/...), envs, and secrets.
- Licenses (proposed): Apache-2.0 for code, CC BY 4.0 for generated data.

- Phase 1 conclusions: closest prior work is Hinglish-TOP (text, Apache-2.0, no desktop domain).
  ASR fine-tune starting point: Oriserve Whisper-Hindi2Hinglish-Swift (whisper-base, Apache-2.0, Roman output).
  Text models to compare: MuRIL joint tagger vs Qwen 0.5B/0.6B LoRA (+ rule/fuzzy baseline).
  TTS teacher: Indic Parler-TTS (Apache-2.0, gated), run on Kaggle.
- lipi-hn is CC BY-NC-SA: use it as an eval baseline only, never build on its weights.

- 2026-11-11: Direction = Bolo-Edge (see RESEARCH.md §6). Edge targets: M2 Mac + Android phone.
  Add Bolo-tiny (from-scratch tiny SLU model) to Phase 3; direct speech→intent as a stretch goal.
- 2026-11-11: Zero-cost project. No paid APIs. Paraphrasing = local MLX model, after templates-first.
  Heavy compute = free Kaggle/Colab.

- 2026-11-11: Splits are by slot VALUE (stable SHA-1 hash, 70/15/15). ~20% of templates are test-only.
  Closed-vocab texts may also appear in train; they're flagged `meta.seen_in_train` rather than dropped
  (dropping starved train of short negatives).
- 2026-11-11: Each row carries `asr_text` (canonical Roman, the ASR target) and `tts_text` (mixed script,
  the TTS input), so the same generator feeds Phase 4.
- 2026-11-11: Per-intent quotas instead of random intent sampling (random sampling under-filled
  closed-vocab intents because of dedup).

- 2026-11-11: MuRIL's last layer has LayerNorm gain ~0.015 (near-constant output). Use the second-to-last layer + mean pool.
- 2026-11-11: mlx-lm completion datasets go through the chat template, so inference must apply it too.
- 2026-11-11: Paraphraser = mlx-community/Qwen2.5-3B-Instruct-4bit (Qwen Research License, research/eval only).
  Model deleted after use (as the user asked). para_* data and the `*_para*` Bolo-tiny models are research-only.
  The recommended release model is `phono_views_aug` (no paraphrases) until paraphrases are regenerated with an Apache-2.0 model.
- 2026-11-11: Paraphrase quality: 56% of test paraphrases kept after review; MuRIL agreed with 74% of train paraphrases.
- 2026-11-11: Negative results kept: consistency loss (−1.4 EM), MuRIL distillation (−4.0 EM).

- 2026-11-11: Phase 4 design. Recorder = stdlib http.server bound to 127.0.0.1 with an inline page. The browser
  resamples to 16 kHz WAV (no ffmpeg on the machine). Browser AEC/NS/AGC are off. Prompts come from the TEST split only:
  80 read + 40 free ("say it your way"). The free prompts are labelled by goal, not transcript.
- 2026-11-11: Synthetic voices split by speaker (test voices never in train). English voices only read Roman text
  of English-heavy commands.
- 2026-11-11: Whisper training uses a plain PyTorch loop (not Seq2SeqTrainer) to be robust to Kaggle's transformers
  version. Decode/train prefix language="en" (as on the Swift card); baselines also run with "hi".
- 2026-11-11: Phonetic WER is reported alongside raw WER/CER (Roman Hinglish has no standard spelling). Devanagari
  hypotheses are transliterated before scoring, which still penalises loanwords (टर्मिनल→tarminal). This is documented.
- 2026-11-11: TTS synthesis code (`tts.synthesize`) is NOT run locally (the model is >2 GB). The notebook starts with a
  small trial N. Known risk: the parler-tts batch API (`audios_length`) is based on its INFERENCE.md.
- Real recordings never leave the laptop (not uploaded to Kaggle). Speech eval in Phase 5 is local.

- 2026-11-11: The user asked Claude to run the Kaggle jobs itself with the existing ~/.kaggle credentials. Everything pushed is
  private. Only synthetic text and code are uploaded, never recordings.
- 2026-11-11: Per-utterance speech-eval hypotheses go to data/eval/ (gitignored). Only aggregates go to results/speech/.
- Known metric artefact: spoken Hindi numbers ("chaalis") vs reference digits ("40") count as WER errors although
  the NLU still extracts 40. The reference is not changed; noted in the results.

- 2026-11-11: Roadmap (user): ship Bolo v0.1 (listen→understand→act) first. Then Phase 6 "Bolo-Awaaz", our own female voice
  distilled from Indic Parler (a voice made from a description only) into a tiny VITS-style TTS. Later, a Jarvis-like Mac agent with Bolo as the fast layer.
- 2026-11-11: Assistant names: **Myna** for the public release ("Hey Myna", a talking bird); **Kiwi** as the user's personal wake word. The wake word is customizable (the detector is trained on synthetic TTS clips of the phrase). "Bolo" is a bad wake word: it's a common Hindi word.
  Wake UX: chime + menu-bar icon, ~8 s silence auto-sleep (configurable), "Kiwi, bas" / hotkey to sleep. Check for name collisions before release.

- 2026-11-11: Safety lock: `scripts/pre-commit` (installed in .git/hooks) refuses commits containing audio, `recordings/`,
  speakers.json, data/eval per-utterance outputs, weights, credentials, HF tokens or private share links. Human recordings and
  their per-utterance transcripts never go to git. Only aggregate numbers do, and per-group numbers need >=3 speakers.
  After a fresh clone: `cp scripts/pre-commit .git/hooks/ && chmod +x .git/hooks/pre-commit`.

- 2026-11-11: `bolo record --share` tested end to end by the user on their own phone over the Cloudflare link: the mic worked,
  audio was saved on the laptop, and "Delete my recordings" removed it from the laptop.

## Open questions
- No real noise corpus yet (MUSAN ~11 GB, ESC-50 NC). Augmentation noise is synthetic for now.
- No LICENSE file yet (pyproject says Apache-2.0). Add it in Phase 7.
- OpenSLR-104 license: OpenSLR says CC BY-SA 4.0, Srota card says CC BY 4.0. Treat as BY-SA.
- HiACC license on Zenodo: not yet confirmed.
- Qwen Research License: do its terms cover outputs (our paraphrases)? Until resolved, treat them as research-only.
- 2026-11-12: Commands expanded 9 -> 20 (+ none). Bolo Tiny v1, 3 seeds on the 2000-row v1 test set: exact match 86.2 ± 0.8%,
  intent 92.6 ± 0.8%, false actions 6.1 ± 1.5%, over-abstain 3.6 ± 1.1%. (The first single run's 2.2% false actions was a
  lucky run; training on the M2 is not bit-for-bit repeatable, so text results are now reported as 3-seed means.)
- 2026-11-12: Confidence cut-off is now chosen as best dev exact match with dev false actions <= 3% (the old rule maximised
  exact match only and sometimes picked 0). It is saved as threshold.json next to the model and the pipeline uses it;
  before this the pipeline always used 0, so shipped behaviour did not match the reported text numbers.
- 2026-11-12: Remaining false actions are confident (~0.9) mistakes on unseen near-miss patterns ("naya folder banao",
  "X ka number batao", "file kholo"); no cut-off can catch them. Fix later with more varied near-miss negatives.
  Tried: more weight on bare "terminal kholo" templates: no change over 3 seeds (EM 85.5%, FAR 6.7%), reverted.
- 2026-11-12: Assistant name is user-settable (`bolo name`, default Myna); the pipeline strips a spoken address.
- 2026-11-12: Step 5 of docs/research/REAL_VOICE_ACCURACY.md done. `phono_views_aug_asr` = Bolo Tiny trained on clean
  text + 9000 rows of Bolo Sun 5-beam transcripts of the 3192 TTS train clips (`scripts/make_asr_noised.py`, slots
  re-located by sound, 1263 rows dropped). 3 seeds each, paired (`scripts/compare_nlu.py`):
  human (1 speaker, 55, locked test): 60.6% -> 72.1% correct action; per seed fixed/broken 5/0 (p 0.06), 8/2 (p 0.11),
  8/0 (p 0.008). macOS voices (240): 83.6% -> 83.8%, acted on none 7.9% -> 9.5% (of 21, i.e. ~0.3 clips). Text test:
  EM 86.2 -> 87.4%, FAR 6.1 -> 5.2%. Kept: it is the new default (`pipeline.DEFAULT_NLU`, seed 0, chosen by convention,
  not by its human score). Step 4 (word-level sound repair) dropped: real errors are closer to other words.
- 2026-11-12: data/say_test_v1 labels fixed: 6 of 27 "none" clips were real commands under the 20-command schema
  (screen lock, bluetooth on x2, screenshot, close system settings / activity monitor); labelled before those commands
  existed. Backup of the old labels: manifest.v0labels.jsonl. Real false actions there are ~2/21, not 7-9/27. Earlier
  macOS-voice numbers for the 20-command set (82.5%) used the old labels.
- 2026-11-12: Bolo Tiny training could crash when augmentation pushed a 40-word row to 41 tags; tags are now cut like words.
- 2026-11-12: Speech v1 retrain prepared: real RIRs (OpenSLR 28, downloaded on Kaggle), ~3% noise-only clips with empty
  targets, 4413 TTS clips (20 commands) as private Kaggle dataset bolo-tts-v1 (`scripts/make_tts_dataset.py`,
  `kaggle_push.py ttsdata`).
- 2026-11-12: Websites are snapped to the known site list using the words around them (`grounding.snap_site`, both
  listen paths): "money control" / a partial "control" -> moneycontrol; unknown sites pass through. Human (seed 0):
  70.9% -> 76.4% (3 clips fixed, 0 broken); macOS voices unchanged (83.75%). Caveat: this fix was found by looking at
  errors on the human set, so 76.4% is optimistic; the next recording session is the clean check.
- 2026-11-12: Remaining human errors (seed 0, after snapping): mishearings (quit arc -> "width arduino", dhundho ->
  "hoga", google -> "go well", ananya -> "peyna"): for the speech v1 retrain. Unseen phrasings ("brightness zyada
  karo", "thoda aur badha", "jo previous chal raha"): not patched from the test set; cover in new templates only if
  a fresh session confirms. 2-3 free-speech labels look wrong (spoken text differs from the prompt): ask the user.
