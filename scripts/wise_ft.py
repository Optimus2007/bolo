"""WiSE-FT weight interpolation (Wortsman et al., 2022): theta = (1 - a) * base + a * finetuned.

Keeps most of the command-domain gain while recovering some of the base model's general robustness.
Usage: python scripts/wise_ft.py --alpha 0.5  ->  models/bolo-whisper-wise050
"""
import argparse

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Oriserve/Whisper-Hindi2Hinglish-Swift")
ap.add_argument("--ft", default="models/bolo-whisper")
ap.add_argument("--alpha", type=float, required=True)
a = ap.parse_args()
base = WhisperForConditionalGeneration.from_pretrained(a.base, torch_dtype=torch.float32)
ft = WhisperForConditionalGeneration.from_pretrained(a.ft, torch_dtype=torch.float32)
sb, sf = base.state_dict(), ft.state_dict()
mixed = {k: ((1 - a.alpha) * sb[k] + a.alpha * sf[k]) if sb[k].is_floating_point() else sf[k] for k in sf}
ft.load_state_dict(mixed)
out = f"models/bolo-whisper-wise{int(round(a.alpha * 100)):03d}"
ft.save_pretrained(out, safe_serialization=True)
WhisperProcessor.from_pretrained(a.ft).save_pretrained(out)
print("saved", out)
