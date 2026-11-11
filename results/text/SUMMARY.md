# Text intent+slot results (synthetic test set, v0)

Mean ± std over seeds (n = number of seeds). All numbers in %. Test = 1,000 rows with unseen slot
values; 'unseen-text' excludes rows whose exact text occurs in train; 'held-out tpl' = phrasings never
seen in training. Latency = single utterance, CPU, p50.

| Model | n | Params | Exact match | EM unseen-text | EM held-out tpl | EM Devanagari | Intent acc | Slot F1 | Slot F1 lenient | Abstain F1 | False-action rate ↓ | AURC ↓ | Script consistency | Latency p50 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| muril_aug | 1 | 237.61M | 91.6 | 90.2 | 72.0 | 86.1 | 95.5 | 91.4 | 92.5 | 94.0 | 3.6 | 5.4 | 88.9 | 15.57 |
| bolo-tiny/phono_views_aug | 3 | 0.69M | 87.9 ± 0.2 | 86.2 ± 0.2 | 59.2 ± 0.8 | 82.1 ± 1.2 | 94.0 ± 0.4 | 87.8 ± 0.1 | 88.9 ± 0.1 | 91.6 ± 1.8 | 6.5 ± 1.4 | 8.4 ± 0.9 | 88.5 ± 0.4 | 0.40 |
| bolo-tiny/phono_cons03_aug | 3 | 0.69M | 86.3 ± 1.0 | 84.3 ± 0.9 | 54.4 ± 2.5 | 81.1 ± 1.8 | 92.9 ± 1.1 | 86.6 ± 0.6 | 87.7 ± 0.6 | 88.4 ± 2.8 | 9.1 ± 0.4 | 9.1 ± 0.3 | 88.6 ± 0.5 | 0.41 |
| bolo-tiny/plain_views_aug | 3 | 0.69M | 85.9 ± 0.8 | 83.9 ± 0.8 | 54.4 ± 0.6 | 81.3 ± 2.0 | 93.0 ± 0.3 | 86.2 ± 1.1 | 87.3 ± 1.2 | 88.2 ± 1.4 | 9.1 ± 0.4 | 8.6 ± 0.2 | 86.9 ± 1.0 | 0.38 |
| bolo-tiny/phono_aug | 3 | 0.69M | 85.2 ± 0.7 | 83.6 ± 0.9 | 55.5 ± 4.0 | 76.7 ± 1.9 | 92.8 ± 0.6 | 85.0 ± 1.0 | 86.2 ± 1.0 | 88.3 ± 2.0 | 8.2 ± 2.3 | 9.7 ± 0.9 | 87.8 ± 0.2 | 0.40 |
| rules | 1 | – | 84.9 | 84.2 | 87.4 | 74.0 | 91.0 | 86.3 | 87.4 | 79.3 | 18.7 | 16.8 | 85.8 | 0.01 |
| bolo-tiny/phono_views | 1 | 0.69M | 82.5 | 79.8 | 53.8 | 76.9 | 90.0 | 83.8 | 84.9 | 82.1 | 4.3 | 10.3 | 88.8 | 0.40 |
| bolo-tiny/plain_consistency | 1 | 0.69M | 79.0 | 75.6 | 48.4 | 78.0 | 87.7 | 81.4 | 82.5 | 73.9 | 6.5 | 13.6 | 88.0 | 0.37 |
| bolo-tiny/phono | 3 | 0.69M | 78.8 ± 2.3 | 75.4 ± 2.6 | 50.7 ± 3.0 | 74.4 ± 5.3 | 88.0 ± 1.0 | 81.3 ± 1.8 | 82.4 ± 1.9 | 75.7 ± 2.4 | 4.1 ± 1.8 | 12.6 ± 1.4 | 89.3 ± 0.5 | 0.41 |
| bolo-tiny/phono_consistency | 1 | 0.69M | 76.1 | 72.3 | 52.8 | 71.7 | 86.7 | 77.8 | 78.9 | 75.4 | 0.7 | 14.2 | 90.6 | 0.41 |
| bolo-tiny/plain | 3 | 0.69M | 73.8 ± 2.3 | 69.6 ± 2.7 | 45.8 ± 2.5 | 65.1 ± 1.2 | 86.3 ± 1.9 | 76.1 ± 2.1 | 77.2 ± 2.1 | 74.6 ± 4.1 | 3.4 ± 1.1 | 16.9 ± 0.9 | 83.8 ± 1.2 | 0.37 |
| qwen0.5b-lora-4bit (MLX GPU latency) | 1 | 494.03M | 56.5 | 51.4 | 34.1 | 37.6 | 77.7 | 58.7 | 58.7 | 56.7 | 15.8 | 21.9 | 54.7 | 198.00 |
| muril | 1 | 237.61M | 13.9 | 11.3 | 15.9 | 13.3 | 13.9 | 0.0 | 0.0 | 24.4 | 0.0 | 90.6 | 100.0 | 19.68 |
