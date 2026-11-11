# Datasheet: Bolo Text Commands v0 (synthetic)

_Follows the structure of "Datasheets for Datasets" (Gebru et al.). Last updated 2026-11-11._

## Motivation
- **Purpose:** train and evaluate small on-device models that map Hinglish (Hindi-English code-mixed)
  desktop voice commands to `{intent, slots}`, including abstaining (`none`) on unclear input.
- **Gap:** to our knowledge, no public Hinglish dataset covers desktop/developer commands (terminal,
  Claude Code, folders, apps, calls, volume/brightness) with slot labels and abstention examples.
  See `RESEARCH.md`.
- **Creator:** the Bolo project. **Funding:** none.

## Composition
- **v0 sizes (seed 13):** train 5,000 · dev 750 · test 1,000 utterances (text only).
- **Intents (9):** `open_terminal`, `open_claude_in_folder`, `call_contact`, `open_app`, `check_status`,
  `open_file_or_folder`, `set_volume`, `set_brightness`, `none`. Schema in `configs/schema.yaml`.
- **Negatives (`none`, ~14%):** out-of-domain (`ood`), unsupported near-misses (`unsupported`, e.g. "terminal
  band karo"), missing required slot (`incomplete`), `ambiguous`, and `filler`/cancel. The reason is in `meta.reason`.
- **Scripts:** ~68% Roman, ~16% Devanagari, ~16% mixed (Hindi words in Devanagari, English in Latin).
  English-only templates are always Roman.
- **Code-mixing density** (`meta.en_ratio` = share of English words among non-slot words): roughly half
  Hindi-heavy (<0.34), a quarter balanced, a quarter English-heavy. Exact counts are in `manifest.json`.
- **Variation sources:** verb forms (kholo / khol do / kholna / kholiye / open kar do…), politeness
  (kijiye, dijiye, kya aap … sakte ho), fillers (yaar, zara, please, umm, bhai), typing variants
  (mein→me/mai, karo→kro, please→plz…), word joining (kar do→kardo), casing and punctuation, number
  words in Hindi/English/digits ("pachaas", "fifty", "50").
- **No personal data.** Contact names are common Indian first names and relations (mummy, bhaiya…), not
  anyone's real address book. Folder and app names are generic.

### Record format (JSONL)
| Field | Description |
|---|---|
| `id`, `split` | e.g. `test-00042`, `test` |
| `text` | Input as a user might type it (script + spelling noise) |
| `script` | `roman` \| `deva` \| `mixed` |
| `intent`, `slots` | Label. `slots` values are canonical lowercase Roman. `level` is an int 0–100; directions are enums |
| `spans` | `[{slot, start, end, value}]` character offsets of each slot in `text` (for taggers) |
| `asr_text` | Canonical lowercase Roman, no noise or punctuation. **Target for ASR fine-tuning** |
| `tts_text` | Clean mixed-script rendering. **Input for the TTS teacher** |
| `meta` | `template_id`, `template_heldout`, `reason`, `fillers` (unlabeled pool values), `en_ratio`, `seen_in_train` |

## Splits (how "unseen" is guaranteed)
- **Open-vocabulary values** (folders, files, contacts, apps) are assigned to exactly one split by a stable
  SHA-1 hash of the value (≈70/15/15). Test utterances only contain folder/file/contact/app names that never
  appear in train or dev. This is enforced by tests.
- **Held-out templates:** for intents with ≥8 templates, ~20% of templates (by hash) appear **only in test**
  (`meta.template_heldout`), to measure generalization to unseen phrasing.
- **Closed vocabularies** (status targets, call apps, numbers, directions) are shared. Dev/test texts that
  also occur verbatim in train are kept but flagged `meta.seen_in_train` (~15% of test). Report metrics on
  the unseen subset too.

## Collection process
Fully synthetic: `bolo generate` expands hand-written templates (`configs/templates.yaml`) with a hand-written
lexicon (`configs/lexicon.yaml`). It is deterministic given seed and config. `manifest.json` records the
seed and SHA-256 of each config file. No LLM paraphrasing in v0 (planned for v1 using a local model).

## Known limitations
- **Templated language.** It is less varied than real speech. Models may overfit template structure. The
  held-out templates and the real-speech eval set (Phase 4) exist to expose this.
- **Hand-written Devanagari** for Hindi words and some slot values. Values without a Devanagari form stay in
  Latin even in "deva" mode (which is also realistic). Spellings were not reviewed by a linguist.
- **Inherent ambiguity:** "X kholo" with an unseen X can be an app or a folder. The generator labels it by the
  pool X came from. At runtime, Bolo should resolve against the user's real apps and folders.
- **Only some semantic coverage:** one-shot commands, no multi-step or contextual follow-ups.
- **Speaker/region bias** cannot be assessed for text. The audio datasheet (Phase 4) will cover speakers.

## Uses
- **Intended:** training/evaluating small intent+slot models, ASR target text, TTS prompts for synthetic speech.
- **Not intended:** measuring general Hinglish NLU ability, or any claim about real-user accuracy without the
  real-speech eval set.

## Distribution and maintenance
- **License:** CC BY 4.0 (data). Generator code: Apache-2.0.
- **Regenerate:** `bolo generate --out data/text/v0 --seed 13`. The full data lives in gitignored `data/`; a
  120-row sample is committed at `samples/text_v0_sample.jsonl`.
- **Versioning:** any config change bumps the dataset version. The manifest hashes identify the exact configs.
