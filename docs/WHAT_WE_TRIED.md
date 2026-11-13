# What we tried, what worked, what failed

This page lists every experiment in Bolo in one place, including the ones that did not work. The numbers come from
`results/` (full tables and caveats) and the dated log in `PROGRESS.md`. Weights for the kept models and the main
rejected ones are on the [weights-v1 release](https://github.com/Optimus2007/bolo/releases/tag/weights-v1)
(`uv run python scripts/get_models.py --all`).

> **Read this first: how far the real-speech numbers go.** The "real voice" set is **55 clips from one speaker** (the
> author), recorded on a laptop mic in a quiet room. At n = 55, ±11 points is roughly the 95% interval. Some later
> fixes (site snapping, two label corrections) were found by looking at errors on this same set, so the 80% figure is
> **optimistic**. Every synthetic-voice number is reported on its own and is never averaged with real speech.

## The headline path on real speech (55 clips, 1 speaker, correct action)

| Step | Correct action | Kept? |
|---|---|---|
| Swift (`Oriserve/Whisper-Hindi2Hinglish-Swift`) as is, with Bolo Tiny | about 51% | baseline |
| Bolo Sun v0: Swift fine-tuned on 4,922 synthetic TTS command clips, beam 5 + rerank | 61.8% (WER 31%) | yes |
| Bolo Tiny also trained on Bolo Sun's own transcripts of TTS clips (realistic ASR errors) | 60.6% → 72.1% (3 seeds, paired) | yes |
| Website names snapped to a known list using the words around them | → 76.4% | yes (optimistic: found on this set) |
| Two labels corrected by the speaker, rescored | 80.6% (3-seed mean) | – |
| Bolo Sun v1: 20 commands, real room impulse responses + noise, noise-only clips | 80.0% (seed 0), WER 23.7% → 18.3% | yes, current default |

Split by command: on the 9 commands the speech model was trained on, v0 got 73%; on the 11 it never saw, 48%.

## Speech recognition (Bolo Sun)

| Tried | Result | Verdict |
|---|---|---|
| Plain `whisper-base` / `whisper-small` decoded as Hindi | 120-180% WER: Devanagari output, loops, invented words | unusable |
| Swift (Hinglish in Roman script) | 29.5% WER on synthetic held-out voices | starting point |
| Full fine-tune on synthetic TTS clips (5 epochs, 18 min on a free Kaggle T4) | synthetic WER 29.5% → 10.6%; macOS voices it never heard: 28.8% → 13.3% | **kept** |
| The same, measured on real general Hinglish speech (OpenSLR 104) | 55.2% → 59.1% WER: it forgets a little | accepted trade-off |
| Encoder frozen | learns less (macOS WER 16.2%), still forgets | rejected |
| 2 epochs instead of 5 | learns less (16.4%) | rejected |
| WiSE-FT 50/50 weight blend with Swift | forgets almost nothing (56.0%) but command accuracy drops to 78.8% | kept as an option, not default |
| float16 weights | identical transcripts, 127 → 100 ms, 290 → 142 MB | **kept** |
| Vocabulary prompt (command words, app names) | WER 13.3% → 12.4% / 11.2%, inside noise but consistent | **kept** (short prompt) |
| Hotword boosting (prefix tree over token ids) | no gain on top of the prompt; strong bonuses hurt (87.5% → 80.0%) | in code, off |
| v1: real RIRs (OpenSLR 28), noise, noise-only clips, 20 commands | real WER 23.7% → 18.3%, action accuracy flat (80.0% both) | **kept** |
| v1 trained 10 epochs | synthetic WER better (14.9% vs 16.2%) but real speech **worse** (76.4%, wrong actions 16.4% vs 10.9%) | **rejected**. Lesson: synthetic WER doesn't predict real use |

## Decoding

| Tried | Result | Verdict |
|---|---|---|
| Greedy | 73.8% correct action (240 held-out macOS-voice commands) | old default |
| Beam 5, top guess | 75.0% | – |
| **Beam 5, rerank the 5 guesses by NLU confidence + grounding** | **82.1%**; paired vs greedy: 23 fixed, 3 broken, 95% CI +4.2 to +12.5 points; needless abstentions 17.4% → 7.0% | **kept** (+70 ms) |

## Command understanding (Bolo Tiny, 0.69M parameters, from scratch)

| Tried | Synthetic exact match | Verdict |
|---|---|---|
| Plain char n-gram hashing | 73.8 ± 2.3 | – |
| + PhonoHash (transliteration + phonetic key + consonant skeleton) | 78.8 (+5.0) | **kept** |
| + slot-value randomisation, word dropout, fillers | 85.2 (+6.4) | **kept** |
| + aligned multi-script views (Roman / Devanagari / mixed) | 87.7 (+2.5) | **kept** |
| + cross-script consistency loss | 86.3 (**−1.4**) | rejected (our own idea; didn't work) |
| + distillation from MuRIL | 83.7 (**−4.0**) | rejected |
| + LLM paraphrases (Qwen2.5-3B) | +12.0 on paraphrased test, −1.2 on synthetic | research only: Qwen Research License, weights **not released** |
| + paraphrases filtered by teacher agreement | no gain | rejected |
| + transcripts of TTS clips by Bolo Sun v0 (realistic ASR errors) | real speech 60.6% → 72.1% | **kept, current default** |
| + transcripts by Bolo Sun v1 of noised copies | real speech 81.2% → 77.0%, one seed broke 7 | rejected: v1 had heard those clips, so its errors weren't realistic |
| More weight on bare "terminal kholo" templates | no change over 3 seeds | reverted |
| Int8 ONNX export | 0.8 MB, no accuracy loss, about 1 ms on 1 CPU thread | **kept** |

Baselines on the same text test:

| Model | Synthetic EM | Paraphrased EM | False actions | Size | Latency |
|---|---|---|---|---|---|
| Hand-written rules | 84.9 | 79.9 | **18.7%** | – | 0.01 ms |
| **Bolo Tiny** | 87.7 | 69.9 | 6.7% | **0.8 MB** | 0.4 ms |
| MuRIL-base fine-tuned | **91.6** | 78.5 | **3.6%** | 950 MB | 15.6 ms |
| Qwen2.5-0.5B LoRA, 4-bit | 56.5 | 56.9 | 15.8% | 276 MB | about 200 ms |

The 0.5B generative model was the weakest. It invented slot values, and 4-bit quantisation cost about 10 points.

## Grounding (does that folder/app really exist here?)

- Adds +5 to +11 points of exact match for free. It mostly fixes transliterated names and app-vs-folder confusion.
- Weak spot: when the spoken thing doesn't exist, it still snaps to a similar one about 29% of the time. Example seen:
  "coding folder" → `recordings`.

## Learning from use (simulated)

An opt-in memory learned aliases for misheard names, a usage prior, and the reranking weights by an online bandit step.
In simulation it brought **no gain beyond noise**. A first version made things worse (72.1%) because it memorised
numbers as if they were mishearings. It stays off until a real person tests it.

## Known failures (not fixed; they need model work and fresh test data)

- English words inside Hindi misheard: quit arc → "width arduino", google → "go well", dhundho → "hoga", previous → "premium".
- "kuch bhi random baat" → calls a contact named "random" (a false action).
- "cpu kitna busy hai" → battery. "vs code kholo" → not understood. "chrome kholo" → asks which site.
- Confident mistakes on near-miss phrasings ("naya folder banao", "X ka number batao"). No confidence cut-off catches them.
- Windows and Linux actions are unit-tested only; not yet run on real machines.

## What we'd do next

1. Record 5+ consenting speakers on budget laptops in fresh sessions. That is the only honest next number.
2. Windows app names and everyday phrasings in the templates, new TTS clips, retrain both models.
3. Personal adaptation on device (LoRA on 50-200 of the user's own clips, kept on the laptop).
4. Regenerate paraphrases with an Apache-2.0 model so that gain can be released.
