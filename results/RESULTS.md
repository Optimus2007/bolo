# Bolo Phase 3 results: text intent + slot models

_2026-11-11. All results on synthetic or LLM-paraphrased text. **No real speech yet** (Phase 4/5). Full table with
every metric: [`results/text/SUMMARY.md`](text/SUMMARY.md). Raw per-run JSON: `results/text/*.json`._

## Test sets
- **Synthetic test** (1,000 rows): template-generated, folder/file/contact/app names **never seen in training**,
  ~18% of rows use phrasings (templates) never seen in training, 3 scripts.
- **LLM-paraphrased test** (144 rows): test rows rewritten by a local 3B LLM, then reviewed one by one. Only rows
  whose label still matched were kept (144 of 256 = 56%). Phrasings were not written by us, so it is the
  fairest check of the rule baseline. The review was model-assisted, **not yet native-speaker verified**, and n is
  small (±~6 points).
- **Environment** for grounding: every folder/file/app/contact in the lexicon (~350 entities) plays the user's machine.

## Headline (exact match = intent and all slots correct)

| System | Size on disk | Synthetic EM | Synthetic EM, grounded | Paraphrased EM | Paraphrased EM, grounded | False-action rate (synthetic) | CPU latency |
|---|---|---|---|---|---|---|---|
| Rule baseline | – | 84.9 | 93.2 | 79.9 | 87.5 | 18.7% | 0.01 ms |
| **Bolo-tiny** (phono_views_aug, 3 seeds) | **0.8 MB int8** | 87.7 ± 0.7 | 95.8 (s0) | 69.9 ± 3.1 | 81.5 | 6.7% | 0.4 ms (1 ms ONNX 1-thread) |
| **Bolo-tiny + paraphrases** (3 seeds) ⚠️ research-only | **0.8 MB int8** | 86.5 ± 0.4 | – | **81.9 ± 1.2** | **90.7** | 9.8% | 0.4 ms |
| MuRIL-base fine-tuned (1 seed) | ~950 MB | **91.6** | **96.4** | 78.5 | 82.6 | **3.6%** | 15.6 ms |
| Qwen2.5-0.5B LoRA, MLX 4-bit (1 seed) | 276 MB | 56.5 | – | 56.9 | 61.8 | 15.8% | ~200 ms (GPU) |

⚠️ The paraphrases came from Qwen2.5-3B-Instruct, whose **Qwen Research License** limits use to research/evaluation.
Models and data derived from it are **not** released under Apache/CC-BY until this is resolved (see PROGRESS.md).

## Ablations for Bolo-tiny (synthetic test EM, mean ± std over 3 seeds)
| Step | EM | Δ |
|---|---|---|
| Plain char-n-gram hashing | 73.8 ± 2.3 | – |
| + **PhonoHash** (translit + phonetic key + skeleton) | 78.8 ± 2.3 | +5.0 |
| + slot-value randomization, word dropout, fillers | 85.2 ± 0.7 | +6.4 |
| + aligned multi-script views | 87.7 ± 0.7 | +2.5 |
| (PhonoHash vs plain, both with views+aug) | 87.7 vs 85.9 | +1.8 |
| + cross-script consistency loss (λ=0.3) | 86.3 ± 1.0 | **−1.4** (negative result) |
| + MuRIL distillation (α=0.5, 1 seed) | 83.7 | **−4.0** (negative result) |
| + LLM paraphrases | 86.5 ± 0.4 synthetic / **81.9** paraphrased | −1.2 / **+12.0** |
| + paraphrases filtered by teacher agreement | 87.3 ± 0.6 / 80.6 ± 1.8 | no gain from filtering |

## What we learned
1. **A 0.69M-parameter model trained from scratch is competitive on device.** With grounding it is within 0.6 EM of
   a 238M-parameter MuRIL on synthetic test (95.8 vs 96.4), at ~1/1000 the size and ~40x lower latency.
2. **Synthetic-only scores overstate quality.** Every learned model loses 10–20 points on LLM-paraphrased phrasings. The
   held-out templates showed the same thing: Bolo-tiny got 59% on unseen templates, MuRIL 72%.
3. **Paraphrase data, even noisy, is the most effective fix** for phrasing generalization (+12 EM). Filtering by teacher
   agreement did not help, and distillation hurt.
4. **Grounding is a large, free gain** (+5 to +11 EM). It mostly fixes Devanagari transliteration of names and the
   app-vs-folder ambiguity. Its weakness: if the spoken entity does not exist, it still snaps to a similar one ~29% of
   the time (abstains 71%). The threshold needs tuning on real data.
5. **The rule baseline is strong on in-domain phrasing** (it was written alongside the templates), but it has the highest
   false-action rate (18.7%), i.e. it acts when it should ask.
6. **The 0.5B generative LLM was the weakest learned option** after 1 LoRA epoch. It hallucinates slot values (अंजलि → "jim")
   and 4-bit quantization costs ~10 EM (63.5 → 53.5 on a 200-row check). More epochs or constrained decoding might help,
   but the latency (~200 ms) and size make it a poor fit for this task anyway.
7. **Two of our novelty ideas did not pan out as designed:** the cross-script consistency loss and MuRIL distillation both
   reduced EM. The multi-script *views* (as plain augmentation) did help.

## Caveats
- Single seed for MuRIL and Qwen (training cost). Bolo-tiny is not bit-for-bit deterministic across runs (seed 0 gave 88.1
  then 86.9 on rerun), hence 3 seeds.
- Latency: Python, M2 Air CPU, batch 1. Phone numbers to come.
- All text is synthetic or LLM-generated. Real Hinglish speech, ASR errors and real users will change these numbers.
