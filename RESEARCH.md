# RESEARCH — Phase 1

_Compiled 2026-11-11 from web searches and direct reads of model/dataset pages. "Verified" means the
license was read on the official page; "unverified" means a secondary source only. Recheck licenses
before release; cards change._

## TL;DR

- **Hinglish ASR exists** (Oriserve Whisper-Hindi2Hinglish family, Srota, lipi-hn, Shunya zero-stt-hinglish),
  but every one targets general/conversational speech. None is evaluated on short command utterances with
  folder/app/contact names.
- **Hinglish text semantic parsing exists**: Google's **Hinglish-TOP** (Apache-2.0, ~10k human + 170k
  synthetic, TOPv2 domains: alarms, reminders, weather, navigation, music, etc.). It is text-only and has no
  desktop/developer-tool domain.
- **Spoken SLU datasets** (SLURP, Fluent Speech Commands, Timers-and-Such) are English. MASSIVE has `hi-IN`,
  but that is translated Devanagari Hindi, not code-mixed, and it is text only.
- **I found no public Hinglish *spoken* desktop/assistant command dataset with intent + slot labels and a
  real-speech benchmark.** That is a claim about what my searches found, not proof that none exists. We
  should keep the wording "to our knowledge" in the README.
- **The gap Bolo fills:** a desktop/developer command domain (terminal, folders, Claude Code, apps, calls,
  system settings), mixed Roman + Devanagari text, unseen slot values at test time, explicit abstention
  ("unclear"), a small human-spoken eval set, and on-device (M-series Mac) latency numbers, all open.

---

## 1. Hinglish speech resources

### 1a. ASR models

| Model | Base / size | Output script | License | Notes |
|---|---|---|---|---|
| [Oriserve/Whisper-Hindi2Hinglish-Swift](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Swift) | whisper-base, 72.6M | **Roman Hinglish** | Apache-2.0 (verified) | ~550 h proprietary noisy Indian-accented audio. Self-reported WER: FLEURS 35.1, CV 38.7, IndicVoices 65.2 (whisper-base: 104–111). **Best fine-tuning starting point for us: small and already Roman-output.** |
| [Oriserve/Whisper-Hindi2Hinglish-Prime](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Prime) / Apex | whisper-large-v3, ~1.5B | Roman Hinglish | Apache-2.0 | Accuracy baseline only. Too big for real-time on a fanless M2; ~3 GB download. ([blog](https://oriserve.com/blog/oriserve-open-sources-india-focused-ai-speech-model-fine-tuned-on-whisper)) |
| [Srota (moorlee/qwen3-asr-0.6b-hinglish)](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish) | Qwen3-ASR-0.6B, ~0.8B | Mixed Devanagari + Latin | Apache-2.0 | Trained on HiACC + OpenSLR-104. Self-reported WER: HiACC 15.9, OpenSLR-104 35.1. Needs transliteration to Roman before we can compare it fairly. |
| [lipi-hn](https://github.com/hitl-systems/lipi-hn) | 14.4 MB 4-bit | Roman | **CC BY-NC-SA 4.0** (non-commercial) | Self-reported Hinglish pWER 29.0. We can use it as an eval baseline only; we can't build on its weights in an Apache/CC-BY release. |
| [shunyalabs/zero-stt-hinglish](https://huggingface.co/shunyalabs/zero-stt-hinglish) | whisper-medium, 0.8B | Mixed Devanagari + Latin | OpenRAIL (use restrictions) | No WER on the card. Baseline only. Shunya's [Pingala](https://huggingface.co/shunyalabs/pingala-v1-universal) is RAIL-M and not Hinglish-specific. |
| [openai/whisper](https://github.com/openai/whisper) tiny/base/small/large-v3-turbo | 39M / 74M / 244M / 809M | Devanagari for Hindi (or translates) | MIT | Plain Whisper gives Devanagari or garbles code-mixing. This is the "off-the-shelf" baseline. |

Apple Silicon: `mlx-whisper` (mlx-community conversions) and whisper.cpp both run tiny/base/small in real
time on an M2. [mlx-tune](https://pypi.org/project/mlx-tune/) claims Whisper LoRA on MLX, but it is
unofficial and I haven't verified it. Full Whisper fine-tuning goes on Kaggle/Colab (PyTorch + HF `Seq2SeqTrainer`).

Cheap win to test before any fine-tuning: **Whisper prompt biasing** (`initial_prompt` listing the user's app,
folder, and contact names). It costs nothing and often fixes proper-noun errors in commands.

### 1b. Speech datasets

| Dataset | Content | Size | License | Use for Bolo |
|---|---|---|---|---|
| [MUCS 2021 / OpenSLR-104](https://openslr.org/104/) | Hi-En code-switched spoken tutorials | 89.9 h train + 5.2 h test | **CC BY-SA 4.0** per OpenSLR (Srota's card says CC BY 4.0, which conflicts; OpenSLR is authoritative) | ASR fine-tune mix (helps general Hinglish acoustics). Share-alike applies to derived data, not to model weights in the usual reading, but flag it. Transcripts are Devanagari+Latin, so we need to transliterate them to Roman. |
| [HiACC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12329218/) (Data in Brief 2025) | Adult + child Hinglish, read + spontaneous | 5.24 h | Zenodo record. License not confirmed by me (Srota says CC BY 4.0) | Small extra ASR eval/train. Check Zenodo before using. |
| [Common Voice Hindi](https://datacollective.mozillafoundation.org/) | Read Hindi | ~544 MB (v25) | CC0-1.0 | Hindi acoustics. It moved to Mozilla Data Collective (account needed) in Oct 2025. |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) | 22 languages, mostly extempore | ~11k h total (gated) | CC BY 4.0 | Hindi subset is large. Download only a slice, on Kaggle. |
| [Svarah](https://github.com/AI4Bharat/Svarah) | Indian-accented English | 9.6 h, 117 speakers | CC BY 4.0 | Eval for the English parts of commands. |
| [Vistaar](https://github.com/AI4Bharat/vistaar) | Benchmark suite (Kathbath, FLEURS, CV, MUCS…) | — | MIT (code); upstream licenses apply | Reuse its eval scripts/normalization conventions. |

### 1c. TTS teachers (for synthetic command audio)

| Model | Size | License | Notes |
|---|---|---|---|
| [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) | 0.9B (~3.6 GB fp32) | Apache-2.0, **gated** (accept on HF) | 21 Indic languages + Indian English, many named speakers, voice described by text prompt, so speaker/style diversity is cheap. **Recommended primary teacher.** Run on Kaggle GPU; on MPS it is slow. |
| [kenpath/svara-tts-v1](https://huggingface.co/kenpath/svara-tts-v1) | 3B (Llama-3.2-3B / Orpheus) | Apache-2.0 on card; base is Llama-3.2, so the **Llama 3.2 Community License likely also applies** | 19 languages, ~50 speakers. Second teacher for diversity, Kaggle only. |
| [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | 0.4B | MIT, **gated**; ToU: clone only voices with permission | Voice cloning from a reference clip. Useful only with consenting reference voices (e.g. your own). |

Risk: TTS teachers read Roman Hinglish poorly (they expect Devanagari for Hindi). The generator should
emit a **Devanagari "speakable" rendering** alongside the Roman target, so the TTS gets clean input while
the ASR label stays Roman. That is one more reason for the normalizer in §4.

---

## 2. Voice-command / SLU datasets

| Dataset | Language | Speech? | Intent+slots? | License | Relevance |
|---|---|---|---|---|---|
| [Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset) ([CST5 paper](https://arxiv.org/abs/2211.07514)) | Hinglish (Roman) | No | Yes (TOP trees) | Apache-2.0 (verified; repo archived 2024) | **Closest prior work.** Good source of natural Hinglish phrasing and a possible extra eval of generalization. Its domains don't include desktop control. [HF mirror](https://huggingface.co/datasets/WillHeld/hinglish_top). |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) | 51–52 locales incl. hi-IN | No | 60 intents, 55 slot types | CC BY 4.0 | hi-IN is Devanagari Hindi, not code-mixed. Includes some `iot_*`/`audio_volume_*` intents we can borrow phrasing from. |
| [SLURP](https://arxiv.org/abs/2011.13205) | English | Yes | Yes | Not verified (check GitHub) | Design reference for the speech→action metric (SLU-F1). |
| Fluent Speech Commands | English | Yes | Yes (action/object/location) | Custom, academic-only | Reference only. |
| [Code-mixed Snips (arXiv 1812.02914)](https://arxiv.org/pdf/1812.02914), CoMTIC ([IEEE 9364402](https://ieeexplore.ieee.org/abstract/document/9364402/)) | Hinglish | No | Intent only | Release status unclear | Prior evidence that code-mixed intent detection is studied, with no usable public data I could find. |
| [COMI-LINGUA](https://aclanthology.org/2025.findings-emnlp.422/), [L3Cube-HingCorpus/HingBERT](https://arxiv.org/pdf/2204.08398) | Hinglish | No | No (LID/POS/NER/MT; raw corpus) | Various | Raw Hinglish text and language ID. HingBERT is a possible encoder alternative. |
| [SynTTS-Commands](https://arxiv.org/html/2511.07821) | En/Zh | Synthetic | Keywords | — | Shows TTS-generated command data works for KWS. Its authors note no OOV/noise eval, which is a lesson for us. |

**Conclusion:** text Hinglish semantic parsing (Hinglish-TOP) and English spoken SLU exist. What I could
not find is a combination of (a) Hinglish, (b) speech, (c) a desktop/dev-tool command domain, (d) an
unseen-slot-value split, and (e) abstention labels. Bolo targets that combination.

---

## 3. Small base models

### 3a. Text intent + slot extraction

| Model | Params | License | Hinglish/Devanagari fit | Mac M2 feasibility |
|---|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** | 0.5B | Apache-2.0 | Multilingual tokenizer, handles Devanagari reasonably | MLX LoRA: minutes per epoch on a few thousand examples; ~1 GB. 4-bit MLX ≈ 300 MB |
| **Qwen3-0.6B** | 0.6B | Apache-2.0 | Similar; strong fine-tuning gains in [Distil Labs' benchmark](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning) | MLX LoRA, same cost |
| SmolLM2-135M / 360M | 135M / 360M | Apache-2.0 (verified for 360M) | Card says English-focused; Devanagari fragments into many tokens | Fastest. Good "how small can we go" data point |
| Gemma 3 270M | 270M | Gemma Terms of Use (not OSI) | Large multilingual vocab | Fast; reports of weak instruction following before fine-tuning |
| **MuRIL** ([google/muril-base-cased](https://huggingface.co/google/muril-base-cased)) | 236M | Apache-2.0 | **Trained with transliterated (Roman) Hindi**, the best fit on paper | Encoder: joint intent classifier + BIO slot tagger, PyTorch MPS, ~5–10 ms/inference |
| IndicBERT v2 ([MLM-only](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only)) | 278M | MIT | Indic + English; less Roman-Hindi exposure than MuRIL | Same as MuRIL |

**Recommendation:** compare **(1) MuRIL joint tagger** (fast and deterministic; confidence from softmax)
against **(2) Qwen2.5-0.5B or Qwen3-0.6B LoRA → JSON** (flexible, easier to extend with new intents via
YAML), with **SmolLM2-360M** as an optional third, cheaper LM. Plus a rule/fuzzy baseline (regex +
`rapidfuzz` over keyword lists).

### 3b. Command ASR

- **Primary:** fine-tune `Oriserve/Whisper-Hindi2Hinglish-Swift` (whisper-base, 73M) on the command domain
  (synthetic TTS + augmentation, plus a slice of OpenSLR-104 transliterated to Roman so it doesn't forget
  general Hinglish). Kaggle T4/P100, a few hours.
- **Secondary:** whisper-tiny (39M) for latency, and whisper-small (244M) if base underfits.
- Small CTC models (IndicWav2Vec, wav2vec2/MMS-style) are possible but need a new Roman-script vocabulary
  head, and MMS weights are CC BY-NC. I'm deferring these unless Whisper latency is a problem.

---

## 4. Romanized Hinglish spelling variation

The problem: one word, many spellings (`kholo/kolo/khollo`, `bahut/bohot/boht`, `karo/kro/karoo`), plus
Devanagari and mixed script in the same utterance.

What others do:
- **Transliteration to a canonical script:** [Aksharantar / IndicXlit](https://arxiv.org/abs/2205.03018)
  (26M pairs, multiple romanizations per word; model ~11M params, MIT; data mixed CC BY / CC0). Roman→Devanagari
  gives one canonical form per word.
- **Phonetic normalization:** [Parikh & Solorio, CALCS 2021](https://preview.aclanthology.org/fix_video/2021.calcs-1.15.pdf)
  (normalization + back-transliteration, G2P); [Yadav et al., ICON 2022](https://preview.aclanthology.org/setup/2022.icon-main.33)
  (phonetics to neutralize variants); Singh et al. 2018 (skip-gram + clustering; cited, not read);
  [Rafae et al., EMNLP 2015](https://preview.aclanthology.org/fix_video/D15-1097.pdf) (UrduPhone + clustering, Roman Urdu).
- **Libraries:** [dhvani](https://pypi.org/project/dhvani/) (IPA-bridged canonicalization; self-reported, unreviewed).

**Bolo approach (Phase 2–3, no model needed yet):**
1. Devanagari → Roman with a deterministic scheme (`indic-transliteration`, simplified ITRANS) so every input is one script.
2. A cheap **phonetic key** (lowercase, collapse doubled vowels, `w→v`, `ee→i`, `oo→u`, drop `h` after
   consonants in the key only, final-schwa `a` optional). Used for keyword/verb matching (`khol*`, `chala*`, `band*`).
3. **Fuzzy matching of slot values** against the user's actual folders/apps/contacts (rapidfuzz on the
   phonetic key). This is how unseen-at-train names get resolved at action time.
4. The generator uses a variant table (`kholo|kholna|khol do|kholdo|kolo`) to inject spelling noise into training text.

Phase 6 (stretch) would replace step 2 with a learned Roman→phoneme/normalized model trained on Aksharantar.

---

## 5. Proposed plan and estimates

Assumptions: M2 Air 16 GB, fanless (keep local runs ≤ ~20 min); Kaggle free tier (T4/P100, ~30 GPU h/week).

| Phase | What | Where | Time (rough) | Disk added |
|---|---|---|---|---|
| 2 | YAML schema, template generator, variant/filler tables, Devanagari renderer, unseen-slot split, datasheet, tests | Mac, CPU, seconds to run | 1–2 sessions | <50 MB (env ~0.5 GB) |
| 2b | LLM paraphrasing: **needs your choice** (see below) | Mac MLX 4-bit, or API | +1 session; ~20–40 min generation | 1–2 GB if local model |
| 3 | Rule/fuzzy baseline; MuRIL joint tagger (MPS); Qwen 0.5B LoRA (MLX); eval; MLX 4-bit export | Mac | 2 sessions; each training run 5–20 min | ~3 GB (torch+mlx env ~2 GB, models ~1.5 GB) |
| 4a | `bolo record` local web tool + consent manifest | Mac | 1 session | recordings ~1 MB/min |
| 4b | Synthetic audio with Indic Parler-TTS + augmentation (noise/speed/reverb) | **Kaggle notebook** | 2–4 GPU h | Download a subset ~1–2 GB to `data/` |
| 4c | Whisper-base (Oriserve Swift) fine-tune notebook | **Kaggle/Colab** | 3–6 GPU h per run | Fine-tuned model ~300 MB (fp16 ~150 MB) |
| 5 | `bolo eval`: WER/CER, intent acc, slot F1, exact match, E2E, latency; baselines (plain Whisper tiny/base/small, Oriserve Swift, Srota, lipi-hn) | Mac (eval only) | 1–2 sessions | Baselines ~4 GB; downloaded one at a time, deleted after |
| 6 | (stretch) G2P/normalizer | Mac | later | <100 MB |
| 7 | README, cards, LICENSE, CONTRIBUTING, CLI polish, CI | Mac | 1 session | — |

Peak project disk: **~8–10 GB**, under the 15 GB cap. Downloads that would exceed 2 GB (and need your OK
first): Indic Parler-TTS weights (~3.6 GB, only on Kaggle, so not on your Mac), Oriserve Prime/Apex
(~3 GB, I'd skip these), and any OpenSLR-104 tarball (~several GB, Kaggle only).

### Decisions I need from you before Phase 2
1. **Paraphrasing LLM:** (a) local Qwen2.5-3B-Instruct 4-bit via MLX (~1.8 GB, free, mediocre Hinglish); (b) a
   paid API (better Hinglish, costs a few dollars, sends only synthetic template text); (c) templates only
   for now. My recommendation: **(a), with human spot-checks**. That keeps everything local and reproducible.
2. **Gated models:** Indic Parler-TTS and IndicF5 need you to accept terms on Hugging Face with your account.
   I won't handle tokens. You'll add `HF_TOKEN` as a Kaggle secret yourself.

### Defaults I'll take unless you object (logged in PROGRESS.md)
- `uv` env with Python 3.12 in `.venv/`. Licenses: code Apache-2.0, data CC BY 4.0.
- Canonical text target = Roman Hinglish (lowercase), with Devanagari accepted as input.
- Test split holds out slot **values** (folders, contacts, apps) entirely, plus a separate small "unclear/negative" set.
