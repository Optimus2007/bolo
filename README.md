# Bolo

**Can a tiny, fully offline model understand spoken Hinglish commands well enough to act on them?**

Bolo is a research project on that question. Commands look like *"downloads folder mein terminal kholo"* and
*"volume 40 pe kar do"*. The aim is to turn speech into the right action on a laptop with no cloud, under ~250 MB, and in
about 0.2 s. This README covers what we built, what we measured, what worked, what failed, and why.

```
"bhai rust learning folder mein terminal kholo"   (speech)
        │  Bolo Sun: Whisper-base fine-tuned on synthetic Hinglish speech (142 MB)    ~180 ms on an M2, 5 guesses
        ▼
5 candidate transcripts
        │  Bolo Tiny: 0.69M-param intent + slot model, built from scratch (0.8 MB)    ~1 ms per guess
        │  grounding: does that folder / app / contact exist on this machine?
        │  rerank: pick the guess that makes the most sense as a command here
        ▼
{intent: open_terminal, folder: "rust learning"}  →  open a terminal in ~/Rust Learning   (or ask, if unsure)
```

## Why it was worth trying

- **People mix languages.** Most people in India talk to devices in mixed Hindi and English, and the technical words
  stay English: *"docker chalu karo"*. In our tests plain Whisper got more words wrong than right on such commands.
  Even a Hinglish-tuned Whisper lost about a third of the correct actions compared with a perfect transcript.
- **No benchmark existed.** To our knowledge, nobody had an open benchmark for Hinglish speech → desktop action.
  The literature search is in [docs/research/BACKGROUND.md](docs/research/BACKGROUND.md).
- **The constraints we set:**
  - zero budget: a free Kaggle T4 plus an Apple M2
  - no real voices in training (privacy)
  - everything on device

## Headline results

Synthetic and real numbers are always reported separately and never averaged.

| Test | Result |
|---|---|
| Text → action, synthetic test with names never seen in training (Bolo Tiny + grounding) | 95.8% exact match (MuRIL-base, 1000× larger: 96.4%) |
| Speech → action, 240 held-out commands in macOS voices (a synthesiser never used in training) | 82.1% (greedy decoding: 73.8%) |
| Speech → action, **real speech: 55 commands from 1 speaker** (laptop mic, quiet room) | **80%** (starting Hinglish Whisper: about 51%) |
| General real Hinglish speech (OpenSLR 104), word error rate | 59.1% vs 55.2% for the base model: slight forgetting |

**How far the real-speech number goes:**
- It comes from one person and 55 clips, so the 95% interval is roughly 68-88%.
- Two fixes and two label corrections were found by looking at this same set, so it is optimistic.
- It is the honest weak point of the project.

## What worked, and why

| Idea | Gain | Why it worked |
|---|---|---|
| **Fine-tune Whisper on synthetic speech only** (Indic Parler-TTS, 4.9k clips, 18 min on a T4) | WER on unseen synthetic voices 29.5% → 10.6%; on macOS voices 28.8% → 13.3% | Command speech is a narrow domain. TTS covers the vocabulary and the Roman-script spelling we want, and the gain carried over to a synthesiser it never heard |
| **PhonoHash features** for Bolo Tiny | +5.0 exact match | Hinglish has no fixed spelling (kholo / kolo / खोलो). Hashing the transliteration, a phonetic key and a consonant skeleton maps the variants close together, with no vocabulary |
| **Slot randomisation, word dropout, fillers** | +6.4 | Stops the model memorising names and exact template wording |
| **Aligned multi-script views** (Roman, Devanagari, mixed) as augmentation | +2.5 | The same command in every script, for free from our generator |
| **Grounding** against the real folders/apps/contacts | +5 to +11 | Fixes transliterated names and app-vs-folder confusion. The model only has to be close; the machine knows the exact name |
| **Beam 5 + rerank by meaning** | +8.3 points (95% CI +4.2 to +12.5), 23 fixed / 3 broken | The right transcript was often the 2nd or 3rd guess. Only the command model and grounding can tell which one makes sense |
| **Train Bolo Tiny on Bolo Sun's own transcripts** of TTS clips | real speech 64.8% → 80.6% (3 seeds, p 0.002-0.03) | Teaches the language model what the speech model actually mishears ("premium" for previous) |
| **Real room echo + noise** in speech training (v1) | real-speech WER 23.7% → 18.3% | Closer to a laptop mic in a real room. Action accuracy stayed flat, though |
| **Vocabulary prompt** to Whisper | WER 13.3% → 12.4% | Small, within noise but consistent, and costs 5-20 ms |
| **float16** and **int8 ONNX** | 290 → 142 MB with identical transcripts; Bolo Tiny 0.8 MB with no loss | – |

## What failed, and why

| Tried | Result | Why (most likely) |
|---|---|---|
| Plain Whisper (base/small, Hindi) | 120-180% WER | It writes Devanagari Hindi, loops on itself and invents words. It was never trained for Roman-script Hinglish |
| **Cross-script consistency loss** (our own idea) | **−1.4** exact match | Forcing identical predictions across scripts over-constrained the model. Devanagari spellings of English names are genuinely noisier, so the loss pulled the clean views toward the noisy ones. The same views as plain augmentation helped |
| **Distillation from MuRIL** | **−4.0** | The teacher's soft labels carried its own errors on unseen names and phrasings, and a 0.69M model spent capacity copying them |
| Qwen2.5-0.5B with LoRA, 4-bit | 56.5% exact match | A generative model invents slot values (अंजलि → "jim"), and 4-bit quantisation cost about 10 points. Also too slow (~200 ms) |
| LLM paraphrases for training | +12 on reworded test | It worked, but the paraphraser (Qwen2.5-3B) is under a research-only licence, so those models are not released |
| Filtering paraphrases by teacher agreement | no gain | Filtering mostly removed the unusual phrasings, which were the useful part |
| Freezing Whisper's encoder / training 2 epochs | learns less, still forgets | The forgetting sits in the decoder (spelling style, vocabulary), not the acoustic encoder. Less training just learns less |
| WiSE-FT 50/50 weight blend | forgets less, but 78.8% vs 82.5% action | A straight trade-off with no free lunch, so it stays an option |
| **Training Whisper 10 epochs** | synthetic WER better (14.9% vs 16.2%), real speech **worse** (76.4% vs 80.0%, more wrong actions) | Overfits the TTS voices. **Synthetic WER is not a reliable guide to real use** |
| Hotword boosting (prefix tree over token ids) | no gain; strong bonus hurt (87.5% → 80.0%) | The vocabulary prompt already gave Whisper the names. A strong bonus forces names into words that aren't names |
| Bolo Tiny on v1's transcripts of noised clips | 81.2% → 77.0% on real speech | v1 had been trained on those very clips, so its mistakes were too few and unrealistic |
| Learning from use (aliases + online bandit on rerank weights) | no gain in simulation; the first version made it worse (72.1%) | Synthetic voices don't repeat a person's habits, and the first version memorised numbers ("volume 80 meant 85") as if they were mishearings |

## What is still wrong

- **English words inside Hindi** are still misheard: quit arc → "width arduino", google → "go well", previous →
  "premium". The likely cause: TTS teachers pronounce English words more cleanly than real speakers do.
- **Unseen phrasings break the command model**:
  - "thoda aur badha do" is not recognised
  - "kuch bhi random baat" → calls a contact named "random"
  - "cpu kitna busy hai" → battery
  
  All of its training text is template-generated.
- **Confident mistakes on near-miss phrasings** ("naya folder banao"). No confidence cut-off catches them.
- **Grounding snaps to a similar name** about 29% of the time when the spoken one doesn't exist.
- **One real speaker.** Everything about real speech rests on 55 clips.

The full tables with every metric are in [docs/WHAT_WE_TRIED.md](docs/WHAT_WE_TRIED.md),
[results/RESULTS.md](results/RESULTS.md) (text) and [results/SPEECH_RESULTS.md](results/SPEECH_RESULTS.md) (speech). The
research behind the speech work is in [docs/research/](docs/research/).

## Reproduce

```bash
git clone https://github.com/Optimus2007/bolo && cd bolo
uv venv && uv pip install -e ".[dev,train,speech]"
uv run python scripts/get_models.py          # trained models (about 220 MB); --all adds the rejected ones
uv run pytest -q

uv run bolo understand "zara battery kitni bachi hai"             # text → intent + slots
uv run bolo understand command.wav --audio                        # speech → intent + slots
```

The models are also on Hugging Face: [kungfupanda12/bolo-sun](https://huggingface.co/kungfupanda12/bolo-sun) (speech)
and [kungfupanda12/bolo-tiny](https://huggingface.co/kungfupanda12/bolo-tiny) (commands).

| Step | Where |
|---|---|
| Generate the synthetic text dataset | `uv run bolo generate` (templates in `configs/`, datasheet in `docs/DATASHEET.md`) |
| Train / ablate Bolo Tiny | `scripts/train_tiny.py`; MuRIL baseline `scripts/train_encoder.py` |
| Make TTS speech, fine-tune Whisper, ablations (Kaggle T4) | `notebooks/01_tts_synth.ipynb`, `02_whisper_finetune.ipynb`, `03_whisper_ablation.ipynb` |
| ASR-error training data for Bolo Tiny | `scripts/make_asr_noised.py` |
| Evaluations | `scripts/eval_*.py`; raw results in `results/` |

Code map: `src/bolo/` holds the dataset generator, models, grounding, decoding and pipeline. It also holds the small
action layer and local app used for the live tests.

## Built on

| Thing | Source | Licence |
|---|---|---|
| Whisper Hindi2Hinglish Swift (base of Bolo Sun) | Oriserve | Apache 2.0 |
| Indic Parler-TTS (synthetic training speech) | AI4Bharat | Apache 2.0 |
| OpenSLR 28 room impulse responses and noise (augmentation) | openslr.org | Apache 2.0 |
| MuRIL, Qwen2.5-0.5B (comparisons only) | Google, Alibaba Qwen | Apache 2.0 |
| Qwen2.5-3B (paraphrases, research only, not released) | Alibaba Qwen | Qwen Research Licence |
| OpenSLR 104 Hindi-English test (evaluation only, not redistributed) | MUCS 2021 | CC BY-SA 4.0 |

Code and models: Apache 2.0.
