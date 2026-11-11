# Bolo speech results (Phase 5)

_2026-11-11. Speech recognition and full speech to action results. Every set is reported on its own; we never average
synthetic and real numbers together. Raw JSON files sit next to this report in `results/speech/` and `results/asr/`._

## The models

| Name | What it is |
|---|---|
| whisper base | `openai/whisper-base`, decoded as Hindi (its natural output is Devanagari) |
| whisper small | `openai/whisper-small`, decoded as Hindi |
| Swift | `Oriserve/Whisper-Hindi2Hinglish-Swift`, a whisper base fine tune that writes Hinglish in Roman script |
| **Bolo Sun** | Swift fine tuned by us on 4,922 synthetic command clips (Indic Parler TTS, train voices only), 5 epochs, 18 minutes on a free Kaggle T4 |
| Bolo Sun blend | WiSE FT: 50% Swift weights and 50% Bolo Sun weights, no extra training |

All five are the same size (whisper base, 73M parameters, about 290 MB) except whisper small (244M).

## 1. Synthetic test, held out voices (977 clips)

Indic Parler TTS, voices Aman, Rani, Priya and Vikas, which never appear in training. Names and phrasings are from the
held out test split. Run on Kaggle.

| Model | WER | CER | Phonetic WER | Exact sentences |
|---|---|---|---|---|
| whisper base (hi) | 180.9% | 61.7% | 179.0% | 9.6% |
| whisper base (en) | 74.8% | 45.6% | 73.1% | 12.4% |
| whisper small (hi) | 81.2% | 45.5% | 75.5% | 0.8% |
| Swift (en) | 29.5% | 13.9% | 25.6% | 25.2% |
| **Bolo Sun** | **10.6%** | **4.8%** | **9.2%** | **59.5%** |

## 2. A different synthesiser: macOS voices (80 clips, full speech to action)

Lekha (hi_IN), Aman, Tara and Rishi (en_IN) reading the read prompts. The model never heard this synthesiser, so this
checks that Bolo Sun did not just learn the quirks of Parler TTS. Action accuracy uses Bolo Tiny with grounding.

| Speech model | WER | Phonetic WER | Correct action | False actions | Median latency (M2) |
|---|---|---|---|---|---|
| perfect transcript (upper limit) | 0 | 0 | 95.0% | 0% | |
| whisper base (hi) | 120.7% | 117.5% | 38.8% | 0% | 131 ms |
| Swift | 28.8% | 24.9% | 62.5% | 0% | 129 ms |
| Bolo Sun blend (50/50) | 19.6% | 16.9% | 78.8% | 0% | 127 ms |
| **Bolo Sun** | **13.3%** | **12.1%** | **82.5%** | **0%** | **127 ms** |

With 80 clips, a difference of a few points is within noise (roughly ±9 points). The large gaps are real.
The gap between speech and a perfect transcript shrank from 32.5 points (Swift) to 12.5 points (Bolo Sun).

## 3. Real human speech: OpenSLR 104 Hindi English test (300 segments, speech recognition only)

Real code mixed narration from spoken tutorials, 56 minutes, CC BY SA 4.0. This is not command speech, so it tests how
much general ability the fine tune kept. The data was downloaded for the run and deleted afterwards.

| Speech model | WER | CER | Phonetic WER |
|---|---|---|---|
| whisper base (hi) | 149.7% | 78.3% | 146.7% |
| **Swift** | **55.2%** | **33.7%** | **44.8%** |
| Bolo Sun blend (50/50) | 56.0% | 33.9% | 45.6% |
| Bolo Sun | 59.1% | 35.4% | 49.0% |

Bolo Sun is about 4 points worse than Swift here. It is not cutting sentences short (output length matches the reference),
so the loss comes from two things: it learned our spelling conventions ("hai" where these references transliterate to
"hain"), and it forgot a little general vocabulary ("Ritre" for writer). The 50/50 blend gets almost all of that back
but gives up some command accuracy. Absolute numbers on this set are inflated for every model because its segment
boundaries are loose (see `results/speech/openslr104_test.md`).

Stored in float16 (142 MB instead of 290 MB), Bolo Sun gives exactly the same transcripts on this set and runs in
100 ms instead of 127 ms (`results/speech/say_fp16.json`).

**Vocabulary prompting** (`results/speech/prompting.json`): giving Bolo Sun a short hint of expected words lowers
WER on the macOS voices from 13.3% to 12.4% (core command words, 75 characters) or 11.2% (command words plus app
names), and correct actions from 82.5% to 86.3% / 85.0%. With 80 clips these gains are inside the noise, but both
point the same way and cost 5 to 20 ms, so the Python API uses the short prompt by default and adds the apps found on
the machine when grounding is on.

## 4. Ablation: can we keep the command gain and forget less?

Two extra variants were trained on Kaggle with the same data (`notebooks/03_whisper_ablation.ipynb`).

| Variant | Synthetic WER | macOS voices WER | Correct action | OpenSLR WER | Train time |
|---|---|---|---|---|---|
| Swift (start) | 29.5% | 28.8% | 62.5% | **55.2%** | |
| **Bolo Sun (full, 5 epochs)** | **10.6%** | **13.3%** | **82.5%** | 59.1% | 18 min |
| Encoder frozen (5 epochs) | 12.5% | 16.2% | 76.3% | 57.2% | 15 min |
| Full model, 2 epochs | 12.8% | 16.4% | 76.3% | 57.5% | 7 min |
| Weight blend 50/50 | | 19.6% | 78.8% | 56.0% | none |

Every way of forgetting less also learns less, and none of them beats the full fine tune on commands, so Bolo Sun stays
the default. One detail is worth noting: freezing the encoder still loses 2 points on general speech, so a good part of
the forgetting sits in the decoder (spelling style and vocabulary), not in the acoustic encoder. A future fix would
mix some general Hinglish text or speech into training, with a licence that allows it.

## What we take from this

1. **Fine tuning on synthetic command speech works, and it carries over to a synthesiser it never heard.** Word errors
   fell by about 55% to 65% and correct actions rose by 20 points on the macOS voices.
2. **There is a real trade off.** The specialised model is slightly worse on long general speech. For a command assistant
   that is the right trade, so Bolo Sun is the default and the blend is kept as an option.
3. **Speech is still the bottleneck.** Even with Bolo Sun, about 12 points of action accuracy are lost between speech and
   a perfect transcript. Remaining errors are mostly rare names and English words inside Hindi sentences
   (docker heard as "doctor", raspberry pi as "rasberry py").
4. **Plain Whisper is not usable for this.** It produces Devanagari, loops on itself and often invents words.
5. **The real test is still missing.** None of the numbers above come from people saying commands. That needs the
   `bolo record` sessions with friends, and those will be the headline numbers.

## Not yet measured

Speech to action on real people saying commands, results by speaker group, phone latency, and any model larger than
whisper base.
