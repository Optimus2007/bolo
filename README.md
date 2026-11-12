# Bolo

Bolo turns spoken Hinglish into an action on your computer, fully offline.

You say *"client folder mein claude code khol do"* and Bolo works out that you want Claude Code started in the folder
called client, checks that the folder really exists on your machine, and hands that back as a small, clear result.
If it is not sure, or the folder does not exist, it says so and asks instead of guessing.

Everything runs on your own laptop or phone. No audio leaves the device, nothing needs the internet, and it costs nothing.

```
"bhai rust learning folder mein terminal kholo"
        │  Bolo Sun, the speech recogniser (142 MB)          about 180 ms on an M2 (5 guesses)
        ▼
"bhai rust learning folder mein terminal kholo"
        │  Bolo Tiny (0.8 MB, built from scratch)            about 1 ms
        ▼
{intent: open_terminal, folder: "rust learning"}
        │  grounding: does that folder exist here?
        ▼
open Terminal in ~/Rust Learning        (or ask, if there is no such folder)
```

> Status: research preview (v0.1). It works and it is measured, but it has been tested on very few real voices so far.
> Please read the limits section before relying on it.

## Why this exists

Most people in India do not talk to their devices in pure Hindi or pure English. They mix, and the technical words stay
in English: *"docker chalu karo"*, *"volume chaalis pe kar do"*, *"terminal kholo"*. Speech tools handle this badly.
In our tests plain Whisper got more words wrong than right on short Hinglish commands, and even a Hinglish tuned
Whisper lost about a third of the correct actions compared with a perfect transcript.

We also could not find an open benchmark that checks the full path from speech to the right action for Hinglish desktop
commands, so Bolo includes one.

Three things we cared about:

1. **Private.** Audio stays on the device.
2. **Small.** The language model part is under 1 MB, so it fits on cheap phones and old laptops.
3. **Careful.** It would rather ask than do the wrong thing. We measure how often it acts when it should not.

## What is in the box

| Part | What it does |
|---|---|
| Dataset generator | Writes thousands of realistic Hinglish commands in Roman, Devanagari and mixed script from YAML templates, with names held out so tests are fair |
| Bolo Tiny | The from scratch understanding model, 0.69M parameters. Uses PhonoHash, a spelling tolerant way of reading Hinglish words |
| Grounding | Matches what you said against the folders, files, apps and contacts that actually exist, and abstains when nothing fits |
| Speech recogniser | A Whisper model fine tuned on synthetic Hinglish command speech, with Roman script output |
| `bolo record` | A consent first recording tool, local or over a private link, for building a real voice test set |
| `bolo eval` | The speech to action benchmark: word error rate, action accuracy, how often it acts wrongly, and speed |

## Results so far

All numbers are our own measurements. Synthetic means generated data; real means recorded from people.

**Understanding text** (exact match: the intent and every detail correct, on names never seen in training)

| Model | Size | Synthetic test | After grounding | Reworded test |
|---|---|---|---|---|
| Hand written rules | none | 84.9% | 93.2% | 79.9% |
| **Bolo Tiny** | **0.8 MB** | 87.7% | **95.8%** | 69.9% |
| MuRIL (fine tuned) | 950 MB | 91.6% | 96.4% | 78.5% |
| Qwen2.5 0.5B (LoRA, 4 bit) | 276 MB | 56.5% | | |

Bolo Tiny ends up within a point of a model a thousand times its size once grounding is on.

**From speech to action** (80 clips spoken by the built in macOS voices, so still synthetic)

| Speech model | Word errors | Correct action | Wrong actions | Time per command |
|---|---|---|---|---|
| Perfect transcript (the ceiling) | 0% | 95.0% | 0% | |
| Plain Whisper base | 120.7% | 38.8% | 0% | 131 ms |
| Hinglish Whisper (Oriserve Swift) | 28.8% | 62.5% | 0% | 129 ms |
| **Bolo Sun (our fine tune)** | **13.3%** | **82.5%** | **0%** | **127 ms (100 ms in float16)** |

**Smarter decoding** (240 different commands, also macOS voices, kept apart from everything we tuned on)

Instead of taking Bolo Sun's single best guess, Bolo now keeps its five best guesses and lets Bolo Tiny and the
grounding step pick the one that makes the most sense as a command on this machine.

| Decoding | Word errors | Correct action | Asks instead of acting when it should not | Time per command |
|---|---|---|---|---|
| Perfect transcript (the ceiling) | 0% | 93.8% | | |
| One guess (before) | 16.0% | 73.8% | 17.4% | 107 ms |
| **Five guesses, picked by meaning (now the default)** | **12.5%** | **82.1%** | **7.0%** | **177 ms** |

That is 23 commands fixed and 3 broken, a gain of 8.3 points (95% interval 4.2 to 12.5), and it closes about 40% of
the distance to the ceiling. It costs about 70 ms. Pass `beams=1` for the old, faster behaviour.
We also tried boosting known app and folder names during decoding; it did not help on top of the vocabulary hint, so
it is off. Details: `results/SPEECH_RESULTS.md`, section 5, and the full story of what we tried, with the research behind it, in `docs/research/IMPROVING_SPEECH_ACCURACY.md`. What the first real-voice test showed, and the plan it led to:
`docs/research/REAL_VOICE_ACCURACY.md`.

On 977 synthetic test clips in voices it never heard in training, Bolo Sun gets 10.6% word errors against 29.5% for
Swift. These clips and the macOS voices are still synthetic.

**Real human Hinglish speech** (300 clips from the OpenSLR 104 tutorial recordings, speech recognition only)

| Speech model | Word errors |
|---|---|
| Plain Whisper base | 149.7% |
| **Hinglish Whisper (Oriserve Swift)** | **55.2%** |
| Bolo Sun | 59.1% |

This one is a fair warning: on long general speech Bolo Sun is a little worse than the model it started from, because it
specialised in commands. A 50/50 weight blend of the two (`scripts/wise_ft.py`) gets most of that back if you need it.

**First real voice** (one person, 55 commands on a laptop mic, quiet room; the speech model has seen 9 of the 20 commands)

| Bolo Tiny trained on | Correct action (mean of 3 seeds) | Wrong action | Asks instead of acting |
|---|---|---|---|
| Clean text only | 64.8% | 15.2% | 21.6% |
| **Clean text + what Bolo Sun actually heard (now the default)** | **80.6%** | **7.3%** | **13.1%** |

The second model is also trained on Bolo Sun's own transcripts of synthetic clips, mistakes included, so it learns
that "premium" can mean "previous". Across the 3 seeds it fixed 28 commands and broke 2 (each seed significant,
p 0.002 to 0.03). It is one speaker and 55 clips, so the 95% interval on 80% is roughly 68 to 88%. Two things
were found by looking at this set, so these numbers are a bit optimistic: website names are now snapped to the known
list ("money control" -> moneycontrol, both rows use it), and two labels were corrected by the speaker (one clip was
chat, not a command; one did name WhatsApp). The next recording session is the clean check. On 240 computer-voice
commands it is unchanged (83.8% vs 83.6%), and on typed text it is no worse (exact match 87.4% vs 86.2%, false
actions 5.2% vs 6.1%). Details: `results/speech/nlu_asr_noised.json`.

The full reports with every metric and caveat live in `results/`.

## Quick start

You need Python 3.10 or newer. We use [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd bolo
uv venv && uv pip install -e ".[dev,train,speech]"
uv run bolo generate                     # builds the synthetic dataset in data/text/v0
uv run pytest -q                         # everything should pass
```

Understand a typed command:

```bash
uv run bolo understand "zara battery kitni bachi hai"
uv run bolo understand "documents folder mein terminal kholo" --ground    # checks what exists on this Mac
```

Understand a recording (a 16 bit WAV file):

```bash
uv run bolo understand command.wav --audio            # uses models/bolo-whisper-fp16 if present
```

From Python:

```python
from bolo.pipeline import Bolo, scan_mac

bolo = Bolo.load(asr="models/bolo-whisper-fp16", env=scan_mac())
bolo.understand("volume 40 pe kar do")
# {'intent': 'set_volume', 'slots': {'level': 40}, 'abstain': False, ...}
bolo.listen("command.wav")
```

### The assistant's name

Bolo is the name of the models. The assistant built on them is called **Myna** by default, and you can call it
anything you like:

```bash
uv run bolo name             # Myna
uv run bolo name Jarvis      # from now on: "Jarvis, volume kam karo"
uv run bolo name --reset     # back to Myna
```

Commands work with or without the name in front ("Myna, wifi band karo", "hey myna screenshot lo",
"volume kam karo, myna", "मैना ..."). The name is matched by sound, so small spelling differences from the recogniser
still count. It is not removed when it is clearly a person you mean ("myna ko call karo" calls a contact named
Myna). A wake word for a custom name ("Hey Jarvis") comes later with the voice assistant; for now the name only
matters inside a command.

Bolo never runs anything by itself. Your program decides what to do with the result, and anything risky should be
confirmed by the user first.

## Recording real voices

```bash
uv run bolo record              # on this laptop
uv run bolo record --share      # gives you a private link friends can open on their phones
```

Each person picks a nickname, gives consent, and reads about 40 short commands (around four minutes), with an optional
bonus round where they say things in their own words. The shared link uses a free Cloudflare tunnel, needs a secret
invite code, closes by itself after two hours, and the audio is stored only on your laptop.

A few rules we keep, and that the code enforces where it can:

1. Recordings are used only to **test** Bolo, never to train it. The training code refuses them.
2. Recordings are never committed to git. A pre commit hook blocks audio, recordings and secrets
   (`cp scripts/pre-commit .git/hooks/`).
3. Anyone can delete their own recordings at any time from the recording page.
4. Nothing becomes public unless that person ticked the public box, and they will be asked again before any release.

Details are in [`docs/AUDIO.md`](docs/AUDIO.md).

## Training the speech model

Training happens on the free Kaggle GPUs, not on your laptop. `notebooks/01_tts_synth.ipynb` makes synthetic command
speech with AI4Bharat's Indic Parler TTS, using different voices for training and testing. `notebooks/02_whisper_finetune.ipynb`
fine tunes Whisper on it with noise, room echo, speed and phone quality augmentation, and compares against plain Whisper and
an existing Hinglish Whisper. `scripts/kaggle_push.py` uploads and runs both as private Kaggle notebooks.

## Limits, honestly

1. **Very few real voices so far.** Most numbers come from synthetic speech. We need recordings from more people,
   with different accents and microphones, before any strong claim.
2. **Synthetic training audio.** The speech model learned from generated voices. Real rooms and real people may
   do worse than the synthetic tests suggest.
3. **Nine kinds of command.** Terminal, Claude Code, calls, apps, status questions, files and folders, volume and
   brightness. Everything else should be refused, and mostly is.
4. **Rewording hurts.** Bolo Tiny drops about 18 points on commands phrased in ways it never saw.
5. **Grounding can still pick a near miss.** When the folder you named does not exist, it sometimes matches a similar one.
6. **Hinglish only** for now. Other Indian language mixes are not covered yet.

## What comes next

1. More real voices, ideally ten people or more, half female and half male.
2. Myna, the assistant built on Bolo: a wake word ("Hey Myna"), a small voice of her own, and replies.
3. Running on Android, with real phone timings.

## Data and models we build on

| Thing | Where it comes from | Licence |
|---|---|---|
| Whisper Hindi2Hinglish Swift (starting point for our speech model) | Oriserve | Apache 2.0 |
| Indic Parler TTS (synthetic training voices) | AI4Bharat | Apache 2.0, gated |
| MuRIL (comparison only) | Google | Apache 2.0 |
| Qwen2.5 0.5B Instruct (comparison only) | Alibaba Qwen | Apache 2.0 |
| Qwen2.5 3B Instruct (made the reworded test, research only) | Alibaba Qwen | Qwen Research Licence |
| OpenSLR 104 Hindi English test set (evaluation only, not redistributed) | MUCS 2021 challenge, via openslr.org | CC BY SA 4.0 |

Bolo's own code is Apache 2.0. The synthetic text dataset is meant to be released under CC BY 4.0.
More detail on each model is in `docs/`.

## Contributing

The most useful thing anyone can give right now is four minutes of their voice through `bolo record`.
New command templates are also welcome: they are plain YAML in `configs/`.
