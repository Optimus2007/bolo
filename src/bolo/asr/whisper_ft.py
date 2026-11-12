"""Whisper fine-tuning + evaluation for Roman-script command-domain Hinglish.

Base: Oriserve/Whisper-Hindi2Hinglish-Swift (whisper-base, 72.6M, Apache-2.0, Roman output; its card
decodes with language="en", task="transcribe"). Targets: our canonical Roman `asr_text`.

Deliberately a plain PyTorch loop (no Seq2SeqTrainer) so it runs on whatever transformers version
Kaggle/Colab ships, and on MPS for tiny local smoke tests.
Augmentation (augment.py) is applied on the fly, so every epoch sees a different room/mic/noise.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np

from ..audio.augment import SR, augment, read_wav, resample
from ..audio.metrics import corpus_scores

BASE = "Oriserve/Whisper-Hindi2Hinglish-Swift"


class HumanAudioError(ValueError):
    pass


def is_human(r: dict) -> bool:
    """Rows from `bolo record` carry speaker consent; TTS rows carry `tts_model`."""
    return "consent" in r or ("speaker_id" in r and "tts_model" not in r)


def load_manifest(path: Path | str, for_training: bool = False) -> list[dict]:
    root = Path(path).parent
    if for_training and "recordings" in Path(path).resolve().parts:
        raise HumanAudioError("refusing to train on recordings/: human recordings are EVALUATION-ONLY (see docs/AUDIO.md)")
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if for_training and is_human(r):
                raise HumanAudioError(f"refusing to train on human recording {r.get('utt_id')}: speakers consented to "
                                      "evaluation only")
            r["_wav"] = str(root / r["audio"])
            rows.append(r)
    return rows


def load_audio(path: str) -> np.ndarray:
    x, sr = read_wav(path)
    return resample(x, sr, SR) if sr != SR else x


def target(r: dict) -> str:
    return r.get("asr_text") or r.get("transcript") or ""


def nonspeech_clip(rng: np.random.Generator, noises: list[np.ndarray] | None, sr: int = SR) -> np.ndarray:
    """1-4 s of room noise / fan / keyboard (a real noise clip if given, else coloured noise), quiet to moderate."""
    from ..audio.augment import coloured_noise, gain

    n = int(sr * rng.uniform(1.0, 4.0))
    if noises and rng.random() < 0.8:
        nz = noises[int(rng.integers(len(noises)))]
        off = int(rng.integers(max(1, len(nz) - n)))
        x = np.resize(nz[off:off + n], n).astype("float32")
    else:
        x = coloured_noise(rng, n, str(rng.choice(["white", "pink", "brown"])))
    x = x / (np.abs(x).max() + 1e-6)
    return gain(x, float(rng.uniform(-45, -15)))


def nonspeech_rows(n: int) -> list[dict]:
    return [{"utt_id": f"nonspeech-{i}", "nonspeech": True, "asr_text": "", "split": "train",
             "tts_model": "noise"} for i in range(n)]


class Collator:
    def __init__(self, processor, language: str = "en", train: bool = False, seed: int = 0,
                 noises: list[np.ndarray] | None = None, aug_p: float = 0.8, rirs: list[np.ndarray] | None = None):
        self.fe, self.tok = processor.feature_extractor, processor.tokenizer
        self.tok.set_prefix_tokens(language=language, task="transcribe", predict_timestamps=False)
        self.train, self.noises, self.aug_p, self.rirs = train, noises, aug_p, rirs
        self.seed, self._worker, self.rng = seed, None, np.random.default_rng(seed)

    def _worker_rng(self):
        import torch.utils.data as tud

        wi = tud.get_worker_info()
        key = wi.id if wi else -1
        if key != self._worker:  # each DataLoader worker gets its own augmentation stream
            self._worker, self.rng = key, np.random.default_rng([self.seed, key + 1])
        return self.rng

    def __call__(self, rows: list[dict]):
        import torch

        if self.train:
            self._worker_rng()
        wavs = []
        for r in rows:
            if r.get("nonspeech"):  # noise-only clip with an empty transcript: teaches "nothing said" -> no words
                x = nonspeech_clip(self.rng, self.noises)
            else:
                x = load_audio(r["_wav"])
                if self.train and self.rng.random() < self.aug_p:
                    x, _ = augment(x, self.rng, self.noises, rirs=self.rirs)
            wavs.append(x)
        feats = self.fe(wavs, sampling_rate=SR, return_tensors="pt").input_features
        batch = {"input_features": feats}
        if self.train:
            ids = [self.tok(target(r)).input_ids for r in rows]
            m = max(len(i) for i in ids)
            lab = torch.full((len(ids), m), -100, dtype=torch.long)
            for k, i in enumerate(ids):
                lab[k, : len(i)] = torch.tensor(i)
            # model prepends decoder_start_token itself when shifting labels -> drop it from labels
            if (lab[:, 0] == self.tok.convert_tokens_to_ids("<|startoftranscript|>")).all():
                lab = lab[:, 1:]
            batch["labels"] = lab
        return batch


def transcribe(model, processor, rows: list[dict], language: str | None = "en", batch_size: int = 32,
               device: str = "cuda", max_new_tokens: int = 64, num_beams: int = 1) -> list[str]:
    import torch

    col = Collator(processor, language=language or "en", train=False)
    out = []
    dtype = next(model.parameters()).dtype
    model.eval()
    for b in range(0, len(rows), batch_size):
        feats = col(rows[b: b + batch_size])["input_features"].to(device, dtype)
        kw = {"task": "transcribe", "max_new_tokens": max_new_tokens, "num_beams": num_beams}
        if language:
            kw["language"] = language
        with torch.inference_mode():
            ids = model.generate(input_features=feats, **kw)
        out += [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]
    return out


def evaluate(model, processor, rows: list[dict], normalizer=None, **kw) -> tuple[dict, list[str]]:
    t0 = time.time()
    hyps = transcribe(model, processor, rows, **kw)
    s = corpus_scores([target(r) for r in rows], hyps, normalizer)
    s["sec_per_utt"] = round((time.time() - t0) / max(1, len(rows)), 4)
    return s, hyps


def train(model, processor, train_rows: list[dict], dev_rows: list[dict], out_dir: Path | str,
          epochs: int = 4, batch_size: int = 32, lr: float = 1e-5, warmup: int = 50, device: str = "cuda",
          fp16: bool = True, seed: int = 0, noises=None, eval_every: int = 200, normalizer=None,
          freeze_encoder: bool = False, max_steps: int | None = None, num_workers: int = 2, rirs=None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    bad = [r.get("utt_id") for r in train_rows + dev_rows if is_human(r)]
    if bad:  # second guard: train/dev must be synthetic only
        raise HumanAudioError(f"{len(bad)} human recordings in train/dev (e.g. {bad[0]}); they are evaluation-only")
    random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.config.forced_decoder_ids = None
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    if freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    dl = DataLoader(train_rows, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    collate_fn=Collator(processor, train=True, seed=seed, noises=noises, rirs=rirs), drop_last=True,
                    persistent_workers=num_workers > 0)
    total = max_steps or epochs * len(dl)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / total))))
    use_amp = fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log, step, best = [], 0, None
    t0 = time.time()
    while step < total:
        for batch in dl:
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 25 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
            if step % eval_every == 0 or step == total:
                s, _ = evaluate(model, processor, dev_rows, normalizer, device=device)
                s["step"] = step
                log.append(s)
                print("dev", json.dumps(s), flush=True)
                if best is None or s["wer"] < best["wer"]:
                    best = s
                    model.save_pretrained(out, safe_serialization=True)
                    processor.save_pretrained(out)
            if step >= total:
                break
    (out / "train_log.json").write_text(json.dumps({"log": log, "best": best}, indent=1))
    return {"log": log, "best": best}
