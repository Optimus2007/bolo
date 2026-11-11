# Speech benchmark (2026-11-11T23:45:49, device=mps)

Sets are reported separately and never pooled. E2E EM = exact action after grounding.

## say_read  (n=80, speakers=4, 2.8 min audio)

| ASR | WER | CER | phon. WER | NLU | intent acc | EM strict | **E2E EM (grounded)** | false-action | ASR ms (med) | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| *oracle transcript* | 0 | 0 | 0 | tiny | 0.925 | 0.887 | 0.950 | 0.0 | – | – |
| models/bolo-whisper-fp16 | 0.1326 | 0.0692 | 0.1213 | tiny | 0.900 | 0.600 | 0.825 | 0.0 | 100.2 | 0.048 |
