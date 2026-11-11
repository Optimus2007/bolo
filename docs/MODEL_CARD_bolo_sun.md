# Model card: Bolo Sun

Bolo Sun is the speech recogniser in Bolo. It listens to a short spoken Hinglish command and writes it out in Roman
script, the way people type Hinglish: *"client folder mein claude code khol do"*. ("Sun" is Hindi for "listen".)

## At a glance

| | |
|---|---|
| Task | Speech recognition for short Hinglish voice commands, Roman script output |
| Base model | `Oriserve/Whisper-Hindi2Hinglish-Swift` (itself a fine tune of `openai/whisper-base`) |
| Size | 73M parameters: 142 MB in float16 (default, same accuracy), 290 MB in float32 |
| Speed | about 100 ms per command on a MacBook Air M2 (MPS, float16), 127 ms in float32, batch of one |
| Decoding | `language="en", task="transcribe"`, the same prefix as the base model |
| Licence | Apache 2.0, the same as the base model |

## Training

| | |
|---|---|
| Data | 4,922 synthetic command clips (short clips; the ones we checked were 1 to 4 seconds) made with AI4Bharat Indic Parler TTS from the Bolo training split |
| Voices | Rohit, Divya (Hindi), Thoma, Mary, Kabir, Sneha, Raghav (English). Test voices never used in training |
| Targets | Bolo's canonical Roman `asr_text` (lowercase, no punctuation) |
| Augmentation | on the fly: speed 0.9/1.0/1.1, synthetic room echo, synthetic noise at 5 to 30 dB SNR, phone band filtering, gain |
| Schedule | AdamW, learning rate 1e-5, 50 warmup steps, cosine decay, batch 32, 5 epochs, fp16, one free Kaggle T4, 18 minutes |
| Selection | best checkpoint by dev WER (289 dev clips) |

No human recordings were used for training, and the code refuses to use them.

## Results

| Test | Swift (start) | Bolo Sun |
|---|---|---|
| Synthetic test, unseen voices (977 clips), WER | 29.5% | **10.6%** |
| macOS voices, never seen synthesiser (80 clips), WER | 28.8% | **13.3%** |
| macOS voices, correct action with Bolo Tiny | 62.5% | **82.5%** |
| Real tutorial speech, OpenSLR 104 (300 segments), WER | **55.2%** | 59.1% |

Full details and caveats: [`results/SPEECH_RESULTS.md`](../results/SPEECH_RESULTS.md).

## Limits

1. **It is a specialist.** It is better than its base model at commands and slightly worse at long general speech.
   For dictation use the base model or the 50/50 blend (`scripts/wise_ft.py`).
2. **Trained only on synthetic voices.** Real microphones, rooms and accents have not been tested on commands yet.
3. **Rare names and English words** are its main errors (docker heard as "doctor").
4. **Spelling follows Bolo's conventions**, which do not always match how other datasets romanise Hindi.
5. It inherits Whisper's habit of occasionally adding a word at the end of very short clips.
