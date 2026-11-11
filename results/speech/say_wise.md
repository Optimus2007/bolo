# Speech benchmark (2026-11-11T23:43:15, device=mps)

Sets are reported separately and never pooled. E2E EM = exact action after grounding.

## say_read  (n=80, speakers=4, 2.8 min audio)

| ASR | WER | CER | phon. WER | NLU | intent acc | EM strict | **E2E EM (grounded)** | false-action | ASR ms (med) | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| *oracle transcript* | 0 | 0 | 0 | tiny | 0.925 | 0.887 | 0.950 | 0.0 | – | – |
| models/bolo-whisper-wise050 | 0.1955 | 0.1048 | 0.1685 | tiny | 0.825 | 0.550 | 0.787 | 0.0 | 127.3 | 0.062 |
| models/bolo-whisper-wise070 | 0.1708 | 0.0889 | 0.1483 | tiny | 0.825 | 0.537 | 0.750 | 0.0 | 128.3 | 0.062 |
