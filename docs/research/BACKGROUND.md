# Background research

_Literature and resource survey done before building Bolo, compiled 2026-11-11 from web searches and direct reads of model/dataset pages. "Verified" means the
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

**Bolo approach:**
1. Devanagari → Roman with a deterministic scheme (`indic-transliteration`, simplified ITRANS) so every input is one script.
2. A cheap **phonetic key** (lowercase, collapse doubled vowels, `w→v`, `ee→i`, `oo→u`, drop `h` after
   consonants in the key only, final-schwa `a` optional). Used for keyword/verb matching (`khol*`, `chala*`, `band*`).
3. **Fuzzy matching of slot values** against the user's actual folders/apps/contacts (rapidfuzz on the
   phonetic key). This is how unseen-at-train names get resolved at action time.
4. The generator uses a variant table (`kholo|kholna|khol do|kholdo|kolo`) to inject spelling noise into training text.

A possible extension: replace step 2 with a learned Roman→phoneme model trained on Aksharantar (not done).

## 5. Methods to borrow, and where Bolo could add something new

### 5a. What the literature offers
| Area | Key work | Takeaway for Bolo |
|---|---|---|
| Tiny on-device text models without a vocabulary | [PRADO (EMNLP 2019)](https://aclanthology.org/D19-1506/), pQRNN ([Google writeup](https://syncedreview.com/2020/09/23/google-approaches-bert-level-performance-using-300x-fewer-parameters-with-extension-of-its-new-nlp-model-prado/)), [pNLP-Mixer (ACL Industry 2023)](https://aclanthology.org/2023.acl-industry.6.pdf), [hashed char n-gram FFNs (Botha et al. 2017)](https://arxiv.org/html/1708.00214), [Subword Semantic Hashing](https://ar5iv.labs.arxiv.org/html/1810.07150) | Projection/hash features + a small encoder reach near-BERT accuracy at 100–300x fewer params with no pretraining. pNLP-Mixer beats pQRNN on MTOP (semantic parsing). Hashing removes the vocabulary, so it handles OOV and misspellings by construction. **Basis of Bolo-tiny.** |
| Romanized-Hindi spelling variation | [SoundexHindi](https://papers.ssrn.com/author=3579322), [Editex variants for Hindi names (2008)](https://preview.aclanthology.org/fix_video/I08-5014.pdf), [Parikh & Solorio 2021](https://preview.aclanthology.org/fix_video/2021.calcs-1.15.pdf), [phoneme representations for ASR-robust NLU (Fang et al. 2020)](https://bohrium.dp.tech/paper/arxiv/2310.12727), [PhonemeBERT](https://arxiv.org/abs/2102.00804) | Plain Soundex drops vowels and fails on Hindi. Hindi-aware phonetic keys + edit distance work better. Phoneme-level inputs make NLU robust to ASR errors. |
| Script / code-mix consistency | ["Neither Here Nor There" (2026)](https://arxiv.org/html/2603.19771) trilingual alignment + consistency metric; [IndicKLAR (2026)](https://arxiv.org/pdf/2605.29637) | Consistency across Hindi / English / code-mixed views is measurable and trainable. Nobody (that I found) does this for tiny on-device SLU, where our generator gives **perfectly aligned multi-script views for free**. |
| Abstention / out-of-scope | [CLINC150](https://arxiv.org/pdf/1909.02027), [energy-based unknown-intent detection](https://ar5iv.labs.arxiv.org/html/2107.12542), [conformal intent classification (2024)](https://arxiv.org/pdf/2403.18973), [selective prediction study](https://arxiv.org/pdf/2203.00211v1.pdf) | Options: an explicit OOS class, a confidence threshold, energy scores, conformal sets. Report risk–coverage curves (AURC), not only accuracy. Most work treats OOS as one bucket. We have **typed negatives** (ood / unsupported near-miss / incomplete / ambiguous). |
| Small LLMs for structured output | [Grammar Pruning (EMNLP 2025)](https://preview.aclanthology.org/setup/2025.emnlp-main.858), ["Constraint Tax" (2026 preprint)](https://www.alphaxiv.org/abs/2605.26128), [constrained decoding in small LLMs (preprint)](https://arxiv.org/pdf/2609.23742) | Constrained JSON decoding fixes format but can increase "valid but wrong" outputs in sub-3B models. So report exact-match, not just JSON validity, and treat the LLM as a comparison, not the default. |
| Distillation for on-device SLU | [Alexa Teacher Model (2022)](https://arxiv.org/pdf/2206.07808v1.pdf), [Qwen-0.5B distillation pipeline (2025)](https://arxiv.org/html/2504.13471v2) | In-domain data matters more than size. A later step: distil Qwen/MuRIL into Bolo-tiny on unlabeled paraphrases. |
| ASR biasing & speech-to-intent | [CB-Whisper](https://arxiv.org/html/2309.09552v3), [Hou et al. Interspeech 2025](https://www.isca-archive.org/interspeech_2025/hou25_interspeech.pdf), [TCPGen-Whisper](https://arxiv.org/html/2509.09197v1), [zero-shot trie biasing](https://web3.arxiv.org/pdf/2508.17796), [Skit-S2I (Indian-accented speech-to-intent)](https://arxiv.org/pdf/2212.13015), [edge Whisper tiny/base](https://arxiv.org/pdf/2502.01649) | Biasing Whisper with the user's own entity list (folders, contacts) improves entity recall. A naive comma-list prompt can hallucinate, so measure it. Whisper-small → SLURP intent ≈ 0.77 acc (Skit-S2I paper), so a tiny direct speech→intent model is a real open question. |
| Synthetic code-mixing | [Pratapa et al. EC theory (MSR)](https://www.microsoft.com/en-us/research/blog/cracking-code-mixing-an-important-step-in-making-human-computer-interaction-more-engaging/), [GCM toolkit (EACL 2021 demo)](https://preview.aclanthology.org/moar-dois/2021.eacl-demos.24.pdf), [CALCS 2021 comparison](https://aclanthology.org/2021.calcs-1.6) | Linguistically constrained switching is principled, but simpler methods are often competitive. Our templates already put switches at phrase boundaries. Revisit for v1 paraphrasing. |

### 5b. Bolo's candidate contributions, as planned before the experiments (outcomes in [WHAT_WE_TRIED.md](../WHAT_WE_TRIED.md))
1. **Phonetic projection embeddings (PhonoHash).** Each token → Devanagari→Roman transliteration →
   a Hinglish-aware phonetic key and a consonant skeleton (kholo/kolo/khollo → same key; karo/kro → same
   skeleton) → hashed character n-grams of raw + key + skeleton into a small Bloom-style table. No vocabulary,
   script-invariant, typo-tolerant. Ablated against plain char-n-gram hashing.
2. **Cross-script consistency training.** The generator realises one utterance as Roman / Devanagari / mixed /
   noisy views with aligned labels. We add a consistency loss (symmetric KL between views' intent and
   tag distributions) and report a **script-consistency score** (does the model give the same action for all views?).
3. **Typed abstention evaluation.** Abstention measured per negative type (out-of-domain vs unsupported near-miss vs
   incomplete vs ambiguous), with risk–coverage / AURC and a "false action rate". A false action (doing the wrong
   thing) is worse than asking again.
4. **Environment-grounded slot resolution.** At action time, slot spans are matched against the user's actual
   folders / apps / contacts by phonetic key + edit distance. If nothing matches, Bolo abstains. The same entity
   list later biases Whisper. Evaluated as strict vs grounded slot accuracy.
5. ASR-noise-aware text training (train on Whisper's real outputs for TTS audio), and a tiny
   direct speech→intent head on the Whisper-tiny encoder.

Honesty note: every ingredient above exists separately in the literature. The claim is the combination for
Hinglish on-device SLU, plus an open benchmark. The results, with ablations and the ideas that
didn't help, are in [WHAT_WE_TRIED.md](../WHAT_WE_TRIED.md).
