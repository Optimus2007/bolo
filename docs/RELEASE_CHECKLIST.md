# Release checklist

Nothing here has been done. Everything is prepared so that the owner can do it in one sitting, when ready.
`python scripts/prepare_release.py` stages the files in `data/release/` (not in git).

## 1. Before anything goes public

1. **Real voices.** Record at least 10 people with `bolo record --share` (ideally 5 female and 5 male, different
   regions and microphones), then run `bolo eval --manifest recordings/manifest.jsonl` and put the human numbers in the
   README. Until then the README must keep saying that the numbers are synthetic.
2. **Names.** Search GitHub, PyPI, Hugging Face and a trademark search for "Bolo", "Bolo Sun" and "Myna".
   When last checked, the PyPI name `bolo` was free and `myna` was taken.
3. **Licences.** Keep the paraphrase data and the `*_para*` models private (Qwen Research Licence). The staging script
   already leaves them out. Credit Oriserve (base model) and AI4Bharat (TTS) in the model cards. They are already there.
4. **Privacy sweep.** Run `git log --all --name-only | sort -u` and confirm there is no audio, no `recordings/`, no tokens.
   The pre commit hook should have made this impossible, but check once by eye.
5. **Consent.** Recordings are never part of the release. If you ever want to publish a voice dataset, ask each
   person again, and include only those with `public_release: true` who say yes a second time.

## 2. Code (GitHub)

1. Read the README once more, top to bottom.
2. Make the repo public in the GitHub settings. This is the owner's decision; tools never do it automatically.
3. Add a short description and topics: hinglish, speech recognition, voice assistant, on device, offline.

## 3. Models and data (Hugging Face)

1. Create the repos under your account: `bolo-sun` (model), `bolo-tiny` (model), `bolocmd-v0` (dataset).
2. Upload each folder from `data/release/` with `huggingface-cli upload <repo> data/release/<folder>`.
3. Optional: a free CPU Space with a microphone demo (Gradio) using Bolo Sun and Bolo Tiny.

## 4. Package (PyPI)

1. `uv build` then check the wheel installs in a clean environment (`pip install dist/bolo-0.1.0-py3-none-any.whl`).
2. `uv publish` with a PyPI token. Consider TestPyPI first.

## 5. Launch

1. A 30 to 60 second demo video in Hinglish: say a command, show the result.
2. Posts: r/developersIndia, r/LocalLLaMA, Hacker News (Show HN), LinkedIn, X. Be plain about the limits; people
   respect honest numbers.
3. Ask for four minutes of voice from anyone who wants to help, through `bolo record --share`.
