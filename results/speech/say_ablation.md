# Speech benchmark (2026-11-12T00:17:12, device=mps)

Sets are reported separately and never pooled. E2E EM = exact action after grounding.

## say_read  (n=80, speakers=4, 2.8 min audio)

| ASR | WER | CER | phon. WER | NLU | intent acc | EM strict | **E2E EM (grounded)** | false-action | ASR ms (med) | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| *oracle transcript* | 0 | 0 | 0 | tiny | 0.925 | 0.887 | 0.950 | 0.0 | – | – |
| models/bolo-whisper-freezeenc | 0.1618 | 0.0831 | 0.1483 | tiny | 0.838 | 0.525 | 0.762 | 0.0 | 122.6 | 0.059 |
| models/bolo-whisper-short | 0.164 | 0.0868 | 0.1483 | tiny | 0.875 | 0.537 | 0.762 | 0.0 | 122.1 | 0.059 |
