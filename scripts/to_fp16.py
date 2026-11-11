"""Save a float16 copy of a Whisper model (same accuracy on our tests, half the size).
Usage: python scripts/to_fp16.py models/bolo-whisper models/bolo-whisper-fp16"""
import sys

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

src, dst = sys.argv[1], sys.argv[2]
WhisperForConditionalGeneration.from_pretrained(src, torch_dtype=torch.float16).save_pretrained(dst, safe_serialization=True)
WhisperProcessor.from_pretrained(src).save_pretrained(dst)
print("saved", dst)
