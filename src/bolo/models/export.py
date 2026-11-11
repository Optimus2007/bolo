"""Edge export for Bolo-tiny: ONNX (Android via ONNX Runtime Mobile) with int8 weight quantization.

EmbeddingBag does not export cleanly, so the exported graph takes a padded id tensor
`ids[int64, (1, L, K)]` (0 = padding; hash ids start at 1) and does an equivalent masked mean.
Featurization (normalization + hashing) stays outside the graph; it is small, deterministic
string code, and is ported alongside the model.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .tiny import BoloTiny


class _Exportable(nn.Module):
    def __init__(self, tiny: BoloTiny):
        super().__init__()
        m = tiny.model
        self.weight = m.emb.weight
        self.proj, self.pos, self.enc, self.norm = m.proj, m.pos, m.enc, m.norm
        self.intent_head, self.tag_head = m.intent_head, m.tag_head

    def forward(self, ids):  # (1, MAX_LEN, K); a word slot with no ids (all 0) is padding
        idmask = (ids > 0).float().unsqueeze(-1)
        e = self.weight[ids] * idmask
        cnt = idmask.sum(2)  # (1, L, 1)
        tok = e.sum(2) / cnt.clamp(min=1.0)
        pad = cnt.squeeze(-1) == 0  # (1, L) True = padding word
        L = ids.shape[1]
        h = self.proj(tok) + self.pos(torch.arange(L).unsqueeze(0))
        h = self.norm(self.enc(h, src_key_padding_mask=pad))
        keep = (~pad).float().unsqueeze(-1)
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.intent_head(pooled), self.tag_head(h)


def padded_ids(tiny: BoloTiny, words: list[str]) -> torch.Tensor:
    """(1, max_len, K) int64; rows beyond the utterance are all-zero (padding)."""
    rows = [tiny.feat.word_ids(w) for w in words[: tiny.tc.max_len]] or [[1]]
    k = max(len(r) for r in rows)
    rows = [r + [0] * (k - len(r)) for r in rows]
    rows += [[0] * k] * (tiny.tc.max_len - len(rows))
    return torch.tensor([rows])


def export_onnx(tiny: BoloTiny, out_dir: Path | str, quantize: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mod = _Exportable(tiny).eval()
    ex = padded_ids(tiny, ["bolo", "folder", "mein", "terminal", "kholo"])
    fp32 = out / "bolo_tiny.onnx"
    torch.onnx.export(mod, (ex,), str(fp32), input_names=["ids"], output_names=["intent_logits", "tag_logits"],
                      dynamic_axes={"ids": {2: "K"}}, opset_version=17,
                      dynamo=False)
    info = {"onnx_fp32_mb": round(fp32.stat().st_size / 1e6, 2)}
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out / "bolo_tiny.int8.onnx"
        quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8)
        info["onnx_int8_mb"] = round(q.stat().st_size / 1e6, 2)
    return info
