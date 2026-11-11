# Speech benchmark (2026-11-11T20:34:28, device=mps)

Sets are reported separately and never pooled. E2E EM = exact action after grounding.

## say_read  (n=80, speakers=4, 2.8 min audio)

| ASR | WER | CER | phon. WER | NLU | intent acc | EM strict | **E2E EM (grounded)** | false-action | ASR ms (med) | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| *oracle transcript* | 0 | 0 | 0 | rules | 0.887 | 0.863 | 0.950 | 0.1111 | – | – |
| *oracle transcript* | 0 | 0 | 0 | tiny | 0.925 | 0.887 | 0.950 | 0.0 | – | – |
| whisper-base | 1.2067 | 0.5025 | 1.1753 | rules | 0.388 | 0.250 | 0.350 | 0.0 | 126.3 | 0.064 |
| whisper-base | 1.2067 | 0.5025 | 1.1753 | tiny | 0.487 | 0.275 | 0.388 | 0.0 | 126.3 | 0.064 |
| swift | 0.2876 | 0.1396 | 0.2494 | rules | 0.575 | 0.438 | 0.588 | 0.1111 | 124.4 | 0.06 |
| swift | 0.2876 | 0.1396 | 0.2494 | tiny | 0.738 | 0.463 | 0.625 | 0.0 | 124.4 | 0.06 |
