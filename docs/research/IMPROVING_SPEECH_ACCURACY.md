# Improving Bolo's speech accuracy: research, experiments and results

This file has two parts. The first is what we actually tried after the research and what happened, including what
went wrong. The second is the research report that came before it, kept as it was written (November 2026), so you can
see the reasoning and the papers behind each choice. All numbers below are on synthetic macOS voices; none are on
real people saying commands yet.

## Part 1. What we tried and what happened

### The starting point

Bolo Sun took one guess per command (greedy decoding) and Bolo Tiny read only that guess. On 240 new commands read by
four macOS voices, kept apart from everything we tuned on, that gave 73.8% correct actions against a 93.8% ceiling
with perfect transcripts.

### 1. Five guesses, chosen by meaning: worked, now the default

Bolo Sun now keeps its five best transcripts (beam search). Each one is read by Bolo Tiny and checked against what
exists on the machine, and Bolo picks the action with the best combined score:

    S = mean log p(transcript) + 0.5 * log p_nlu(intent) + 2 * grounding score - 1 * [abstains]

Guesses that lead to the same action add their probabilities together. The three weights were picked on the old 80 clip
set and then checked once on the 240 new commands.

| 240 held out commands | Word errors | Correct action | Gives up when it should act | Median time |
|---|---|---|---|---|
| perfect transcript (ceiling) | 0% | 93.8% | | |
| one guess (before) | 16.0% | 73.8% | 17.4% | 107 ms |
| five guesses, top one only | 15.0% | 75.0% | 16.0% | 177 ms |
| **five guesses, chosen by meaning** | **12.5%** | **82.1%** | **7.0%** | **177 ms** |

23 commands fixed and 3 broken: a gain of 8.3 points with a bootstrap 95% interval of 4.2 to 12.5. This closes about
40% of the gap to the ceiling, which is what the research predicted ("a third to half"). Most of the gain is Bolo
giving up less often: the right reading was often the second or third guess, and only the grounding step could tell.
Beam search alone barely helps; the gain comes from letting meaning choose. The right action was somewhere in the five
guesses 88.3% of the time, so there is still about 6 points to win from better choosing.

### 2. Boosting known names during decoding: did not help, left off

The research ranked this near the top (40 to 60% fewer errors on boosted words in English papers). We built a prefix
tree over the token ids of every app, folder, file and contact name and gave a bonus to tokens that continue a name.
On our data it added nothing on top of the vocabulary prompt Bolo already uses, and strong bonuses hurt: with a bonus of
4 the top guess fell from 87.5% to 80.0% correct on the tuning set, because the model started bending ordinary words
into names. Our best guess at why: the prompt already carries the names, and our test sentences use few rare names per
command. It stays in the code (`TrieBoost`), switched off.

### 3. Learning from use: no gain yet, opt in

`bolo.adapt.Memory` learns from the user's feedback, from text only, never audio. It remembers misheard names, gives a
small bonus to names the user really uses, and adjusts the reranking weights after each accept or correction with one
policy gradient step (a contextual bandit). We simulated one user: the 240 commands, then the same commands again in
other voices, acting first and learning after each one.

| | first pass | second pass |
|---|---|---|
| no learning | 81.7% | 80.8% |
| everything on | 82.1% | 81.3% |

**What went wrong.** The first version made things clearly worse: 72.1% on the second pass. It memorised numbers and
fixed choices as if they were mishearings ("volume 80" corrected to 85 became a rule that 80 means 85, "cpu" corrected
to bluetooth became an alias). We limited aliases to name slots and to pairs that sound alike, which removed the harm,
but no real gain is left. The simulation is weak evidence, because synthetic voices do not have a person's habits, so
Memory stays opt in until a real user tries it.

### 4. A bug we had to work around

In transformers 5, Whisper's built in beam search with several returned sequences gave five copies of the same text and
no scores. We wrote a small beam search of our own (`bolo.asr.decode.beam_search`, with a key value cache), which also
made boosting and scoring easier to control.

### What is next

1. Real voices: run all of this on friends' test recordings (evaluation only) and see if the 8 point gain holds.
2. Training side, from Part 2: replay of general Hinglish speech with a KL anchor to fix forgetting, real room echo and
   noise recordings, and more varied Parler voices.
3. Close the remaining choosing gap (88.3% in the five guesses against 82.1% chosen): a small learned scorer trained on
   synthetic data only.

Code: `src/bolo/asr/decode.py`, `src/bolo/adapt.py`, `scripts/eval_decoding.py`, `scripts/eval_adapt.py`.
Numbers: `results/speech/decoding_tune.json`, `decoding_test.json`, `adapt_sim.json`, and `results/SPEECH_RESULTS.md`.

## Part 2. The research report (written before the experiments)

### Original title: Fix the decoder's search before retraining Bolo

Bolo Sun's biggest losses happen after the acoustic model has done its job, and that part costs nothing to fix. With perfect transcripts the pipeline gets 95% of actions right. With Bolo Sun's transcripts it gets 82.5%. Most of that 12.5-point gap comes from one-best decoding that never looks at the phone's live app, file and contact lists, and from an NLU that sees one guess when it could see five. Beam search with trie hotword boosting, N-best reranking by NLU plus grounding, phonetic entity matching and a calibrated abstain rule are all training-free and on-device, and each fits the 100-200 ms budget. In English and in production assistants, each has delivered double-digit relative gains. The second bottleneck is data, not size. Bolo Sun learned from 4,922 clips of one TTS family. Fine-tuning made real Hinglish worse (59.1% vs 55.2% WER on OpenSLR-104), and every ablation tried so far (frozen encoder, 2 epochs, WiSE-FT 50/50) only slides along the same trade-off curve. Methods that add new information are what can push that curve outward: replay of real open speech with a KL anchor to the base model, real room impulse responses and noise, more varied TTS voices, and pseudo-labels from a bigger Hinglish teacher. Scaling is not the only option. Under the latency limit, scaling parameters is not available at all, and scaling synthetic data hits diminishing returns quickly. The honest caveat is that almost none of the published numbers below were measured on Roman-script Hinglish or on Whisper-base, so treat every expected gain as a direction, not a forecast.

### Bolo loses points between the audio and the action, not only inside the model

Start with where the errors live. Five things limit Bolo Sun today, and they matter in roughly this order.

**1. The pipeline throws away the information it needs.** Bolo decodes a single best transcript, then hands that one string to Bolo Tiny and the grounder. The thing the user cares about is a rare entity like "docker" or an app name. Those are exactly the tokens a narrow fine-tune gives little prior probability to, so a similar-sounding common word ("doctor") wins the single best slot. The correct word is often still sitting in the second to fifth beam. Amazon's Alexa team found the right transcript often sits at ranks 2-5, and feeding the n-best list into intent classification gave **14-25% relative accuracy gains** ([Li et al., arXiv 2001.05284](https://arxiv.org/pdf/2001.05284)). Bolo's 95 to 82.5 drop is a normal pipeline gap, not a bug. On SLURP, a home-assistant benchmark, intent accuracy fell from 84.8 with gold transcripts to 76.7 and 73.4 with two real ASR systems, and entity span F1 fell from 78.2 to 62.7 ([Bastianelli et al., arXiv 2011.13205](https://arxiv.org/pdf/2011.13205)). The known remedies for that gap should therefore apply.

**2. The training audio looks nothing like a friend talking into a phone.** Seven Indic Parler voices, one TTS family, clean read speech and synthetic echo is the hardest case in the synthetic-data literature. Purely synthetic training leaves a gap to real data, and that gap is larger for attention encoder-decoder models like Whisper than for older hybrid systems ([Hilmes et al., arXiv 2407.17997](https://arxiv.org/html/2407.17997v1)). The 13.3% WER on macOS voices measures TTS-to-TTS transfer. It is probably optimistic for humans, who pause, hesitate, mumble and sit in echoey rooms. A June 2026 study of Whisper on Hindi, Kannada and Telugu used IndicParler among its TTS systems. It reportedly found synthetic speech worked best layered on top of real recordings, cutting Hindi WER by about 15%, and that IndicParler ranked behind IndriTTS as a data source ([arXiv 2606.17662](https://arxiv.org/pdf/2606.17662)). Those figures come from a search summary, not the full text.

**3. The decoder forgot its general knowledge.** Here are Bolo's own ablations.

| Variant | OpenSLR-104 WER (general) | Command WER (macOS voices) |
|---|---|---|
| Base (Oriserve Swift) | 55.2% | 28.8% |
| Bolo Sun, full FT, 5 epochs | 59.1% | 13.3% |
| Frozen encoder | 57.2% | 16.2% |
| 2 epochs | 57.5% | 16.4% |
| WiSE-FT 50/50 | 56.0% | 19.6% |

Each variant buys back about 0.5-0.65 points of general WER for every point of command WER it gives up (1.9 for 2.9, 1.6 for 3.1, 3.1 for 6.3). That consistency means all three are points on one trade-off curve, and none beats it. Freezing the encoder recovered only about half of the 3.9 points lost, so much of the damage is in the decoder. The decoder is effectively a language model, and 4,922 narrow command transcripts pulled its word prior and spelling habits toward the command set. The same pattern is well documented for small Whisper models. Whisper-base fine-tuned on Uyghur lost the ability to transcribe Chinese entirely under full fine-tuning, LoRA, AdaLoRA and AdaLoRA with a frozen encoder alike, while large-v2 kept it ([Xu et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/xu24h_interspeech.pdf)).

**4. The ruler is bent.** Roman Hinglish has no standard spelling ("hai/he", "nahi/nahin/nai", "karo/kro"). The project notes that much of the OpenSLR-104 gap is spelling convention rather than mishearing. Automatic spelling-pair normalisation, applied to both reference and hypothesis, reduced measured WER by **13.28% on average** across four languages, and the normalised scores matched human judgements of quality better ([arXiv 2303.05046](https://arxiv.org/abs/2303.05046), abstract only). A worse problem is that WER can improve while the words that matter get worse. Fine-tuning whisper-large-v3 improved Fisher Spanish-English WER by 5% relative, but error on the embedded-language words got **39% relative worse** ([PIER, arXiv 2501.09512](https://arxiv.org/html/2501.09512v2)). For Bolo, the words that matter are app names, file names and the action verb.

**5. Model size is fixed by latency.** Whisper-small (244M) is about 3.3 times larger than base ([Radford et al., arXiv 2212.04356](https://ar5iv.labs.arxiv.org/html/2212.04356)), and turbo and large are far outside a 100-200 ms phone budget. Size can only enter Bolo indirectly, through a teacher.

### The maths underneath is mostly adding scores and comparing distributions

Every method in this report rests on four ideas. You can follow them with nothing beyond logs, sums and averages.

**Decoding is a search for the highest score.** Whisper writes text one token at a time. The probability of a whole sentence is the product of each token's probability given the audio and the tokens before it. Taking logs turns that product into a sum: score(y) = Σ log p(token_t | earlier tokens, audio). Greedy decoding grabs the best token at each step and never looks back. Beam search keeps the B best partial sentences alive at each step and so can recover when the locally best choice turns out wrong. Whisper's default is beam 5 ([arXiv 2309.10299](https://arxiv.org/html/2309.10299v1)). For fine-tuned small Whisper models, beam 5 and greedy differed by 3-17% average WER ([Whisper-LM, arXiv 2503.23542](https://arxiv.org/html/2503.23542v1)).

**Biasing is adding a bonus to that sum.** Bayes' rule says p(text | audio) is proportional to p(audio | text) times p(text). A fine-tuned Whisper has a prior p(text) baked in from its training transcripts. For Bolo, that prior is narrow. Adding λ to the log-score of a token is the same as multiplying its probability by e^λ. A bonus of 2 nats multiplies it by about 7.4. That tips close calls toward "docker" when "docker" is on the phone, and does nothing when the audio clearly says something else. Shallow fusion is the same move with a whole language model: Q = log P_asr + α·log P_LM + β·(word count). Here α sets how much the LM counts and β stops the LM from preferring short outputs ([arXiv 2503.23542](https://arxiv.org/html/2503.23542v1)). A hard grammar sets the bonus for illegal tokens to minus infinity. A soft grammar just subtracts a penalty. N-best reranking adds terms after decoding: S(h) = log p_asr(h) + λ_nlu·log p_nlu(intent, slots | h) + λ_g·grounding(h). The cleanest version groups hypotheses by the action they lead to and adds their probabilities: p(action | audio) ≈ Σ_h p(action | h)·p(h | audio).

**Confidence is about whether probabilities mean what they say.** If the model says 90% across many utterances, it should be right about 90% of the time. Temperature scaling divides the logits by one number T before the softmax, p = softmax(z/T), and fits T on held-out data. T > 1 softens an overconfident model ([Guo et al., arXiv 1706.04599](https://arxiv.org/abs/1706.04599)). A model fine-tuned on clean TTS is likely to be overconfident on real voices. Whisper is reasonably calibrated on clean input but becomes overconfident under noise ([arXiv 2509.07195](https://arxiv.org/html/2509.07195v1)). Abstention then becomes a simple rule: act only when estimated P(correct action) exceeds a threshold τ. τ is chosen from a risk-coverage curve, which plots the error rate on executed commands against the fraction executed ([arXiv 2603.21172](https://arxiv.org/pdf/2603.21172)).

**Training is minimising cross-entropy, and forgetting is what that ignores.** Fine-tuning minimises CE = -Σ log p(correct token) on the command clips only. Nothing in that loss rewards keeping general Hinglish working. Gradients reuse whatever weights help, including those that stored the general vocabulary and spelling. There are three ways to push back:

- **Replay** puts general data back into the loss.
- **KL or LwF** adds β·KL(p_base || p_student). KL is a measure of how different two probability distributions are, so this term pays the student to keep its full next-token distribution close to the old model's. That includes its "dark knowledge" about alternative spellings ([Li & Hoiem, arXiv 1606.09282](https://arxiv.org/abs/1606.09282)).
- **EWC** adds Σ (λ/2)·F_i·(θ_i - θ*_i)², a spring on each weight. The spring is stiff where the Fisher information F_i says the old task is sensitive ([Kirkpatrick et al., arXiv 1612.00796](https://arxiv.org/abs/1612.00796)). L2-SP is the same spring with every F_i equal to 1 ([arXiv 1802.01483](https://arxiv.org/abs/1802.01483)).

WiSE-FT is different. It averages weights after training: θ = (1-a)·θ_base + a·θ_ft. That only works because a fine-tune stays in the same low-loss valley as its starting point, a property called linear mode connectivity ([Frankle et al., arXiv 1912.05671](https://arxiv.org/abs/1912.05671); [WiSE-FT, arXiv 2109.01903](https://arxiv.org/abs/2109.01903)). Bolo's 50/50 blend still working is direct evidence that this holds for Bolo. Interpolation can only pick a point on the line between two models, though, which is why it cannot beat the trade-off curve.

### Scaling helps slowly, and more synthetic data is the weakest kind of scale

Scaling is real but slow. Across Whisper's training languages, log WER falls linearly with log hours, and **WER halves for every 16 times more data** ([arXiv 2212.04356](https://ar5iv.labs.arxiv.org/html/2212.04356)). As a formula, WER ∝ D^(-k) with 16^(-k) = 0.5, so k = log 2 / log 16 = 0.25. Doubling the data multiplies WER by 2^(-0.25) ≈ 0.84, a 16% cut. Going from 4,922 clips to about 80,000 might roughly halve command WER, but only on the synthetic distribution the extra clips come from. The synthetic-data literature says the curve flattens sooner and stops at a higher floor on real speech. Generating about as much TTS data as real data is cost-effective, and more brings marginal gains ([arXiv 2410.16726](https://arxiv.org/pdf/2410.16726)). Synthetic data beat real data only when the TTS itself was trained on more data than the real set ([Minixhofer et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/minixhofer25_interspeech.pdf)). A Québec French study saw raw synthetic data stop helping beyond a 50% share. TTS clips convolved with room impulse responses kept improving up to 100% synthetic ([arXiv 2508.21631](https://arxiv.org/html/2508.21631)). Variety and realism beat volume.

Scaling parameters is ruled out on device. Larger models do keep helping multilingual and low-resource speech ([OWLS, arXiv 2502.10373](https://arxiv.org/abs/2502.10373)), but the only way to borrow that size under 100-200 ms is distillation from a large teacher into base. Distil-Whisper gives the shape of the payoff. Its average out-of-distribution WER went from 13.8 at 435 h of pseudo-labelled audio, to 9.2 at 1,742 h, to 7.8 at 3,483 h, and then plateaued ([Gandhi et al., arXiv 2311.00430](https://arxiv.org/html/2311.00430)). The same paper found the encoder carries the robustness. Halving it cost 3.1 WER points, while cutting the decoder to 2 layers cost little. That is useful later if Bolo ever wants a small encoder with a 2-layer decoder.

So the answer to "is scaling the only option?" is no. The ranking for Bolo is: smarter decoding, then better measurement, then better data distribution (real acoustics, many voices, real open speech), then distillation. Raw synthetic volume comes last.

### Decoding-time methods attack the 12.5-point action gap without any training

**Beam search with trie hotword boosting from the live app list** is the first thing to build. The idea is a prefix tree of the tokenised names currently on the device (apps, files, contacts, plus tech words). During beam search, a hypothesis earns a small bonus for each token that continues a name in the tree. If it then leaves the tree, the bonus is taken back, so half-matches cannot cheat. On Whisper large-v3, this training-free trie cut biased-word WER on LibriSpeech test-clean from 9.86 to 6.11, while unbiased WER barely moved (1.94 to 1.87). Adding filtered pronunciation variants got biased-word WER to 5.66, about 43% relative. Unfiltered variants were catastrophic, sending unbiased WER to 21.19 ([Liu et al., arXiv 2508.17796](https://arxiv.org/html/2508.17796v2)). A 2026 comparison found context biasing cut biased-word WER by up to 88% relative with other words largely unaffected ([Huber & Waibel, arXiv 2608.05759](https://arxiv.org/abs/2608.05759), abstract only). NVIDIA's phrase-boosting tree for its encoder-decoder models costs about 11-20% over greedy ([TurboBias, arXiv 2508.07014](https://arxiv.org/pdf/2508.07014)). Trie boosting complements Bolo's existing prompt_ids biasing, which the project notes credit with taking command WER from 13.3% to 11.2%. Prompting changes what the decoder conditions on, while the trie changes how the search scores. Prompts are also the riskier of the two: the CB-Whisper authors warn that plain comma-separated entity prompts can induce hallucination ([arXiv 2309.09552](https://arxiv.org/abs/2309.09552v1)).

There are two practical traps. First, BPE tokenisation differs with and without a leading space, so tokenise each name in its mid-sentence form. Hugging Face's `sequence_bias` silently fails otherwise ([HF generation docs](https://huggingface.co/docs/transformers/v4.38.0/main_classes/text_generation)). Second, nobody has published trie-biasing latency for Whisper-base on Apple Silicon, so measure it. The encoder runs once and a command is 10-20 tokens. Beam 4-5 batched through a 6-layer decoder with a KV cache should cost tens of milliseconds, but that is an estimate.

**N-best reranking with NLU and grounding** is the method most directly aimed at the 82.5-vs-95 gap, because it only changes which hypothesis wins. For each of the 3-5 beams, run Bolo Tiny and the grounder. Score each beam by length-normalised ASR log-prob plus NLU confidence plus a grounding term, with a large penalty when a required slot matches nothing on the phone. Then sum scores across beams that lead to the same action. Bolo Tiny is small, so five runs should cost a few milliseconds. The Alexa result above (14-25% relative) is the closest production evidence. One caveat: n-best helps in proportion to hypothesis diversity more than oracle WER ([Sahu et al., NeurIPS 2022 workshop](https://mlanthology.org/neuripsw/2022/sahu2022neuripsw-diversity)). Whisper beams tend to be near-duplicates, so pairing reranking with trie boosting matters, because boosting pushes the right entity into the list. My own estimate, not a published number, is that these two together could close a third to a half of the 12.5-point gap. That depends on how often the correct entity appears anywhere in the beam, which you can measure in an afternoon as "oracle action accuracy over the n-best".

**Phonetic entity matching in the grounder** catches what the decoder still misspells ("rasberry py" for "raspberry pi"). Index each app, file and contact three ways: normalised characters, a Hinglish phonetic key, and a Double-Metaphone-style key for English names. Then take the best similarity. One voice-assistant system that combined character-level and phoneme-level retrieval for names reported a **40.8% accuracy improvement** ([Raghuvanshi et al., EMNLP 2019](https://aclanthology.org/D19-3011/)). A non-autoregressive contextual spelling corrector reached up to 51% relative WER reduction over its ASR ([Wang et al., arXiv 2203.00888](https://arxiv.org/abs/2203.00888)). That is a heavier option, and a lexicon matcher gets much of the benefit at zero cost. When a user confirms a correction, store the misheard form as an alias. That is the self-healing idea in [EMNLP 2024 Industry](https://aclanthology.org/2024.emnlp-industry.1.pdf), and it uses the user's own phone at runtime, not friends' recordings in training.

**A spelling-normalised, entity-focused metric** does not change the model, but it changes which experiments look like wins. Collapse reference and hypothesis through the same phonetic key, report raw WER beside it, and add a PIER-style error rate on slot tokens and the action verb. Do not pivot through Devanagari. IndicXlit's Hindi top-1 is only about 59-69% ([Aksharantar, arXiv 2205.03018](https://arxiv.org/pdf/2205.03018); [third-party card](https://huggingface.co/udgamlabs/Lekhani)), and the round trip would Hindi-fy the English words. MUCS 2021's transliterated WER shows the field already treats script choice as non-error ([arXiv 2203.16578](https://arxiv.org/pdf/2203.16578)). Vimarsha's "lattice of variations" scores against several valid spellings ([arXiv 2609.24199](https://arxiv.org/abs/2609.24199), abstract only). Check that your key does not merge distinct command words like "kholo" and "khol".

**Calibrated abstain and confirm** converts the remaining errors from wrong actions into "did you mean X?". Build a tiny logistic regression over these features:

- minimum word probability (the best word-level score per the C-Whisper study, [arXiv 2502.13446](https://arxiv.org/html/2502.13446v1))
- token entropy
- the margin between beams 1 and 2
- whether the n-best agree on the action
- no-speech probability
- intent probability
- catalogue match score and margin

Temperature-scale first. Pick thresholds for a target false-action rate, and report coverage at 1% and 2% selective risk rather than only ECE. ECE measures calibration, not decision usefulness ([arXiv 2509.01455](https://arxiv.org/html/2509.01455v1)). A three-way policy of execute, confirm or reject recovers coverage cheaply in a voice interface.

**Soft grammar and n-gram fusion** come after these. whisper.cpp already ships GBNF grammar decoding with a soft penalty rather than a hard mask, used with penalties of 10 for an assistant grammar and 100 for chess ([whisper.cpp PR #1229](https://github.amrom.workers.dev/ggml-org/whisper.cpp/pull/1229)). A KenLM 5-gram fused into fine-tuned Whisper-base cut WER by 45% (Basque) and 24% (Spanish) in-distribution. The same fusion made Spanish worse out of distribution for Whisper-small (-8%) ([arXiv 2503.23542](https://arxiv.org/html/2503.23542v1)). Bolo's fine-tune already absorbed the command templates into its internal LM, so the marginal gain is uncertain and the over-templating risk on open slots (message text, search queries) is real. Keep α small, leave open slots unconstrained, and treat a big gap between constrained and free decoding scores as an out-of-grammar signal.

**Train Bolo Tiny on what it will actually see.** This is text-only training and needs no new audio. Decode the TTS training clips with Bolo Sun and train the NLU on the ASR outputs with gold labels. Add errors from a confusion-matrix simulator built from aligned ASR-vs-reference pairs ([Wang et al., arXiv 2006.05635](https://arxiv.org/pdf/2006.05635)), and add spelling-variant augmentation (hai/hain, nahi/nahin). Building the simulator from synthetic or public-data alignments keeps it within the friends rule. NLU robustness absorbs a lot: in SLURP's error analysis, about 3,490 entities had ASR errors that the NLU still recovered ([arXiv 2011.13205](https://arxiv.org/pdf/2011.13205)).

### Training methods only pay off when they add new information

The ablation table makes the rule clear: freezing, stopping early and interpolating only trade along the curve. To beat it, training has to see something new about general Hinglish, or about real acoustics.

**Replay with a KL anchor, then a WiSE-FT sweep, is the best-supported forgetting fix.** In a study comparing merging (Linear, TIES, DARE), LoRA-scale discounting and replay, experience replay was "the most effective method", the only one robust in every setting, and better still when combined with others. Their replay ratio was tiny (0.5%), and the replay data was generated by the original model itself ([Hsiao et al., arXiv 2505.17496](https://arxiv.org/abs/2505.17496)). For Bolo:

1. Mix 20-50% real general Hinglish or Hindi audio into each batch.
2. Label it with the base Oriserve model's own outputs. Its outputs follow the base model's spelling convention, which makes replay behave like LwF.
3. Add β·KL(p_base || p_student) using one extra forward pass of the frozen 74M base.
4. Sweep WiSE-FT α on a real dev set.

Extrapolating from the current 18-minute run, this should take under an hour per run on a T4, so 20+ ablations fit inside the 30 h/week quota. That is an estimate, not a measurement. Skip LoRA as a forgetting fix, because base forgot catastrophically even with LoRA in the Uyghur study above. Expect EWC to underperform here. In the Irish/Welsh Whisper study, EWC barely protected a closely related language because the two tasks used the same high-Fisher weights ([Song et al., arXiv 2407.06800](https://arxiv.org/html/2407.06800v2)). Command Hinglish and general Hinglish share almost everything.

**Real RIRs, real noise, codec simulation and SpecAugment** make the synthetic clips sound like the world:

- Swap synthetic echo for real room impulse responses from OpenSLR 28, listed as Apache 2.0 ([OpenSLR 28](https://openslr.org/28)).
- Add MUSAN noise and babble, CC BY 4.0 overall, with per-file licences ([OpenSLR 17](https://www.openslr.org/17)).
- Add Opus, AMR or MP3 re-encoding and 8 kHz resampling with ffmpeg. No paper quantifies this for TTS-trained ASR, but it costs nothing.
- Turn on SpecAugment through the Hugging Face config ([Park et al., arXiv 1904.08779](https://arxiv.org/abs/1904.08779)). The original Whisper models used no augmentation at all, and large-v2 added SpecAugment ([arXiv 2212.04356](https://ar5iv.labs.arxiv.org/html/2212.04356)).
- Widen speed to 0.8-1.25 and insert pauses and fillers ("umm", "haan"). Matching TTS duration statistics to real ones measurably helped ASR in RWTH's work ([arXiv 2310.08132](https://arxiv.org/pdf/2310.08132v1.pdf)).

**Richer TTS** means many voices, not more clips per voice. Parler is description-conditioned, so sample hundreds of speaker descriptions (pitch, pace, gender, mic quality, "distant", "noisy") instead of seven fixed voices. The 2606.17662 study explicitly varied the number of distinct voices ([arXiv 2606.17662](https://arxiv.org/abs/2606.17662)). A second TTS family reduces single-engine overfitting, but check each engine's licence first. If you move the macOS voices into training, you lose your current held-out proxy, so first carve out a set of macOS voices that is never trained on.

**Use one romanisation convention for every training target.** Run every training target through a single deterministic converter, so Whisper-base always writes "hai" and "nahi". Inconsistent targets spend capacity on a coin flip that no one can predict ([discussion in the Oriserve and Trelis model comparisons](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Swift)).

**Mix in openly licensed real speech, transliterated to Roman.** All of the following are Devanagari, so labels need romanisation (or teacher pseudo-labels, below):

- Kathbath: about 150 h Hindi read speech from 1,218 contributors ([HF](https://huggingface.co/datasets/ai4bharat/Kathbath)).
- IndicVoices: CC BY 4.0, diverse read and extempore speech ([HF](https://huggingface.co/datasets/ai4bharat/IndicVoices)).
- SPRING-INX: CC BY 4.0 ([IITM](https://asr.iitm.ac.in/dataset)).
- Svarah: Indian-accented English, CC BY 4.0, useful for "volume", "WiFi" and "alarm" ([HF](https://huggingface.co/datasets/ai4bharat/Svarah)).

OpenSLR-104 is the only real code-mixed source. It is lecture speech under **CC BY-SA 4.0** ([OpenSLR 104](https://openslr.org/104/)), and share-alike may reach trained weights. Also, training on its train split while reporting its test split is fine, but say so. Start with a ladder of 0, 1, 3 and 10 h at a 1:1 to 1:3 real-to-synthetic ratio. Favour many speakers over many hours per speaker.

**Distillation from a bigger Hinglish teacher** is the way to borrow scale. Pseudo-label short real clips with a large Hinglish Whisper. Keep only clips where the teacher agrees with a second model (uDistil-Whisper's proxy-WER filter beat no filtering by 5-7 WER points, [arXiv 2407.01257](https://arxiv.org/html/2407.01257v5)), or whose output parses into Bolo's command grammar. Then train base with Distil-Whisper's 0.8·KL + 1.0·CE loss. In Distil-Whisper, a 10% WER filter took average WER from 13.4 to 11.4 ([arXiv 2311.00430](https://arxiv.org/html/2311.00430)). The same pseudo-labels double as replay data. Labelling 100 h on a T4 is roughly 2-5 GPU-hours, an unverified estimate.

**TCPGen** is the light-training upgrade to trie boosting. It is a small pointer module over the biasing trie, gated against the frozen Whisper. With Whisper base.en it cut rare-word WER from 72.5 to 52.3 on SLURP and from 71.3 to 41.8 on DSTC2 ([Sun et al., arXiv 2306.01942](https://arxiv.org/html/2306.01942)). It was trained on real speech, not TTS, so the transfer to Bolo is uncertain. Try it only after training-free boosting has plateaued.

Several things are not worth the effort now. Test-time adaptation (SUTA, SGEM) gives about 3 points on child speech, at the cost of per-utterance gradient steps on the phone ([SGEM summary](https://chatpaper.com/paper/60498)). Full end-to-end SLU loses the transcript the grounder needs, and Whisper fine-tuned for SLU is mixed on SLURP ([arXiv 2305.09652](https://arxiv.org/pdf/2305.09652)). Moving to Whisper-small might fit with int8, but that is unverified on a phone.

### A ranked plan for zero budget, a T4 and 100-200 ms

The ranking weighs expected gain against cost. Gains are drawn from the cited studies, and almost none of them are Hinglish or Whisper-base, so the last column states how confident I am that the gain transfers.

| Rank | Method | Expected gain (source context) | Cost | Latency | Transfer confidence to Hinglish |
|---|---|---|---|---|---|
| 1 | Spelling-normalised WER + slot/verb error rate + speaker-block bootstrap | No accuracy change; ~13% lower measured WER from normalisation alone; stops you choosing wrong | 1 day, M2 | none | High |
| 2 | Beam 4-5 + trie hotword boosting from live app/file/contact list | 40-60% relative on biased words in English Whisper | 2-4 days, M2 | +tens of ms (measure) | Medium; tokenisation of Roman Hinglish names is the risk |
| 3 | N-best reranking with Bolo Tiny + grounding, summed per action | 14-25% relative intent accuracy (Alexa) | 2-3 days | +few ms | Medium-high; depends on beam diversity |
| 4 | Phonetic entity matching (char + Hinglish key + Metaphone) with learned aliases | 40.8% name-resolution gain in one system | 2 days | <1 ms | Medium; no validated Hinglish Metaphone exists |
| 5 | Calibrated abstain/confirm (temperature + logistic regression) | Fewer wrong actions at fixed coverage; not more correct ones | 1-2 days | <1 ms | High for the method; thresholds must be re-tuned per model |
| 6 | Train Bolo Tiny on ASR outputs + simulated errors + spelling variants | Recovers part of the pipeline gap | 1 day, M2 or T4 | none | Medium-high |
| 7 | Soft grammar + small-α n-gram on fixed command frames | 24-45% in-distribution for base; can hurt OOD | 2-3 days | small | Low-medium; internal LM already knows the templates |
| 8 | Single romanisation convention for all targets, retrain | Cleaner learning, less "error" that is only spelling | 1 run | none | High |
| 9 | Real RIR + MUSAN + codec + SpecAugment + wider prosody | Largest cheap acoustic lever per the Québec and RWTH results | 1-2 T4 runs | none | Medium-high |
| 10 | Replay (pseudo-labelled by base) + KL-to-base, then WiSE-FT sweep | Moves the general-vs-command frontier outward | 5-20 T4 runs | none | Medium; best-evidenced, but not on Whisper-base |
| 11 | Many Parler descriptions + a second TTS family | Coverage gains; IndicParler ranked below IndriTTS in one study | 1-3 T4 sessions to generate | none | Medium |
| 12 | 1-10 h open real speech, transliterated | Real+synthetic beats synthetic in every multi-paper result | several T4 runs | none | Medium; mostly pure Hindi, not code-mixed |
| 13 | Distillation / pseudo-labels from a large Hinglish teacher, agreement-filtered | Big real-speech robustness gains at 100-500 h | 1-2 weeks of quota | none | Medium; teacher's own Hinglish quality caps it |
| 14 | TCPGen | 28-41% relative rare-word WER on base.en | 1 week T4 | small | Low-medium; trained on real speech in the paper |

Rows 1-5 come first because they are reversible, need no training, run on the M2 today, and attack the 12.5-point action gap directly. Row 1 has to come before everything else. Without it, a trie bonus that fixes "docker" but changes "nahi" to "nahin" could look like a regression.

**Licence notes for rows 9-13.**

| Source | Licence | Caveat |
|---|---|---|
| MUSAN | CC BY 4.0 | Attribution; per-file licences apply |
| OpenSLR 28 | Apache 2.0 | Its noises are drawn from MUSAN |
| Kathbath | CC0 or CC BY (catalogues disagree) | Both permissive; HF gated, free click-through |
| IndicVoices | CC BY 4.0 | HF gated, free click-through |
| Svarah | CC BY 4.0 | HF gated, free click-through |
| SPRING-INX | CC BY 4.0 | None noted |
| IndicVoices-R | CC BY 4.0 on HF, CC BY-ND on AIKosh | Avoid unless the conflict is resolved |
| Shrutilipi | CC BY 4.0 | Underlying All India Radio broadcast rights unaddressed |
| OpenSLR-104 | CC BY-SA 4.0 | Release Bolo weights under CC BY-SA, or keep it out of training |

Sources: [data notes on OpenSLR, HF and AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/indicvoices_r.html).

**The friends rule shapes the evaluation, not just the training.** A logistic-regression abstainer, a trie bonus, a fusion α or a WiSE-FT α all count as fitted parameters. The strict reading of "evaluation only" says none of them should be fitted on friends' audio. The cleanest setup tunes on public real speech (OpenSLR-104 dev, Lahaja, Svarah) plus a set of held-out synthetic voices with real RIRs. Friends' audio is then touched only to report final numbers. If a few scalars have to be set on real command audio, split the friends by speaker into a small dev group and a test group. Report only on the test speakers, and label the dev use openly. For the recordings themselves:

- Each friend reads the same fixed list of 50-100 commands, which allows paired comparisons.
- Each adds 10-20 free paraphrases to catch template overfitting.
- Record in two or three logged conditions (quiet, TV on, phone at arm's length).

For the statistics, use a bootstrap that resamples speakers, not utterances. Utterances from the same speaker have correlated errors, so resampling utterances gives intervals that are far too narrow ([Liu et al., arXiv 1912.09508](https://arxiv.org/pdf/1912.09508); [Bisani & Ney 2004](https://www-i6.informatik.rwth-aachen.de/~bisani/bootlog.html)). With about 10 speakers, only differences of several points that hold for most speakers are conclusive. Treat smaller ones as directional. Lahaja (132 speakers, [HF](https://huggingface.co/datasets/ai4bharat/Lahaja)) and Svarah give the accent coverage that 10 friends cannot.

### Conclusion

The shift in thinking is that Bolo is not mainly short of model capacity or clips. It is short of context at decode time and of realism and diversity at training time. The phone already knows which apps and files exist, and that knowledge is the strongest prior available. Today it is applied after transcription, when it should shape the search. Feeding it into beam search and reranking turns a 74M model's near misses into correct actions, without touching the latency class or the training rule. On the training side, the ablations have already shown that more tuning of the same data only moves along one curve. The next real gain needs data the model has never seen: real rooms, many voices, open real speech and a teacher's soft targets.

What remains genuinely unknown is how much of this transfers. No published trie-biasing, n-best or fusion result exists for Roman-script Hinglish, for Whisper-base, or for a model trained only on TTS. That is why the normalised, entity-focused metric and a speaker-level bootstrap are worth building first. With roughly ten evaluation speakers, a careful measurement is the only way to tell a real gain from a lucky one.
