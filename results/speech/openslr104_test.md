# Real human Hinglish speech: OpenSLR-104 Hindi-English test (ASR only)

300 segments (stratified round-robin over 30 recordings, seed 0), 56 min of real code-switched speech
from spoken tutorials (CC BY-SA 4.0). Device: M2 (MPS), batch 1. Script: `scripts/eval_openslr104.py`.

| ASR | decode lang | WER | CER | phonetic WER | median ms / segment | RTF |
|---|---|---|---|---|---|---|
| openai/whisper-base | hi | 1.497 | 0.783 | 1.467 | 156 | 0.029 |
| Oriserve Whisper-Hindi2Hinglish-Swift | en | **0.552** | **0.337** | **0.448** | 156 | 0.026 |

Caveats (read before quoting these numbers):
- **Loose segmentation.** Segment boundaries in this dataset fall on whole seconds (subtitle timings), so audio
  often contains words from the neighbouring segment that are not in the reference. Both models output such
  words (e.g. "...main menu"). This inflates WER for *every* model: use the table for comparison, not as absolute quality.
- **Script mismatch.** References are mixed script and often write English words in Devanagari (ऑफिस, स्लाइड).
  Swift outputs Roman, so raw WER penalises it. Phonetic WER (after transliteration) is the fairer column but
  still imperfect (ऑफिस -> "ophis" vs "office").
- whisper-base (hi) falls into repetition loops on several segments (e.g. "पाप पाप पाप ..."), which explains WER > 1.
- The domain is tutorial narration, not short commands. This measures robustness to real Hinglish voices only;
  command (speech -> action) accuracy needs command recordings (`bolo record`).
- The data was deleted after this run (re-download: 443 MB, see the script docstring). The fine-tuned
  `bolo-whisper` will be added to this table once it is trained.
